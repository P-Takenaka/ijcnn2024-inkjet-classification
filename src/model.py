import torch.nn as nn
import torch
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from .metrics import F1Score

class MLPModule(pl.LightningModule):
    def __init__(self, num_printer_model_classes, optimizer, input_size, hidden_sizes=[], activation_fn=nn.ReLU, dropout_pct=0.0, output_size=None, calculate_metrics=True, **kwargs):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn
        self.dropout_pct = dropout_pct
        self.output_size = output_size
        self.num_printer_model_classes = num_printer_model_classes
        self.optimizer_config = optimizer
        self.calculate_metrics = calculate_metrics

        if self.calculate_metrics:
            self.train_model_f1 = torch.nn.ModuleList([
                torchmetrics.F1Score(task='multiclass', num_classes=self.num_printer_model_classes),
                F1Score(top_k=2),
                F1Score(top_k=3),
                F1Score(top_k=4),
                F1Score(top_k=5)])
            self.train_precision = torchmetrics.Precision(task='multiclass', num_classes=self.num_printer_model_classes)
            self.train_recall = torchmetrics.Recall(task='multiclass', num_classes=self.num_printer_model_classes)


            self.val_model_f1 = torch.nn.ModuleList([
                torchmetrics.F1Score(task='multiclass', num_classes=self.num_printer_model_classes),
                F1Score(top_k=2),
                F1Score(top_k=3),
                F1Score(top_k=4),
                F1Score(top_k=5)])

            self.val_precision = torchmetrics.Precision(task='multiclass', num_classes=self.num_printer_model_classes)
            self.val_recall = torchmetrics.Recall(task='multiclass', num_classes=self.num_printer_model_classes)

            self.test_model_f1 = torch.nn.ModuleList([
                torchmetrics.F1Score(task='multiclass', num_classes=self.num_printer_model_classes),
                F1Score(top_k=2),
                F1Score(top_k=3),
                F1Score(top_k=4),
                F1Score(top_k=5)])

            self.test_precision = torchmetrics.Precision(task='multiclass', num_classes=self.num_printer_model_classes)
            self.test_recall = torchmetrics.Recall(task='multiclass', num_classes=self.num_printer_model_classes)

        self.save_hyperparameters(logger=False)

        if hidden_sizes:
            layers = []
            prev_hidden_size = input_size
            for i in range(len(hidden_sizes)):
                layers.append(nn.Linear(prev_hidden_size, hidden_sizes[i]))
                layers.append(activation_fn())
                prev_hidden_size = hidden_sizes[i]
            if dropout_pct > 0:
                layers.append(nn.Dropout(p=dropout_pct))
        else:
            prev_hidden_size = input_size
            layers = []

        self.feature_extractor = nn.Sequential(*layers)
        self.printer_model_out_layer = nn.Linear(prev_hidden_size, self.num_printer_model_classes)

    def configure_optimizers(self):
        params_list = filter(lambda p: p.requires_grad,
                                 self.parameters())

        assert(self.optimizer_config is not None)
        optimizer = optim.Adam(params=params_list, lr=self.optimizer_config['lr'],
                               weight_decay=self.optimizer_config['weight_decay'] if 'weight_decay' in self.optimizer_config else 0)

        return {'optimizer': optimizer}

    def log_training(self, outputs):
        assert(self.calculate_metrics)
        # Necessary here and not above due to data parallel considerations
        self.log('train/loss', outputs['loss'], on_epoch=True, on_step=True,
                     prog_bar=True, sync_dist=True)

        for k, v in outputs['loss_dict'].items():
            self.log(f'train/{k}', v, on_epoch=True, on_step=False,
                     prog_bar=False, sync_dist=True)


        if self.trainer.is_global_zero:
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'],
                     rank_zero_only=True)

        for i in range(len(self.train_model_f1)):
            self.train_model_f1[i](preds=outputs['logits_model'], target=outputs['target_model'])
            self.log(f'train/model/f1' if i == 0 else f'train/model/f1_top{i+1}', self.train_model_f1[i],
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        self.train_precision(preds=outputs['logits_model'], target=outputs['target_model'])
        self.log(f'train/model/precision', self.train_precision,
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        self.train_recall(preds=outputs['logits_model'], target=outputs['target_model'])
        self.log(f'train/model/recall', self.train_recall,
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)


    def log_validation(self, outputs):
        assert(self.calculate_metrics)
        # Necessary here and not above due to data parallel considerations
        self.log('val/loss', outputs['loss'], on_epoch=True, on_step=False,
                     prog_bar=True, sync_dist=True)

        for k, v in outputs['loss_dict'].items():
            self.log(f'val/{k}', v, on_epoch=True, on_step=False,
                     prog_bar=True, sync_dist=True)

        for i in range(len(self.val_model_f1)):
            self.val_model_f1[i](preds=outputs['logits_model'], target=outputs['target_model'])
            self.log(f'val/model/f1' if i == 0 else f'val/model/f1_top{i+1}', self.val_model_f1[i],
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        self.val_precision(preds=outputs['logits_model'], target=outputs['target_model'])
        self.log(f'val/model/precision', self.val_precision,
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        self.val_recall(preds=outputs['logits_model'], target=outputs['target_model'])
        self.log(f'val/model/recall', self.val_recall,
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)


    def log_test(self, outputs):
        assert(self.calculate_metrics)
        # Necessary here and not above due to data parallel considerations
        self.log('test/loss', outputs['loss'], on_epoch=True, on_step=False,
                     prog_bar=True, sync_dist=True)

        for k, v in outputs['loss_dict'].items():
            self.log(f'test/{k}', v, on_epoch=True, on_step=False,
                     prog_bar=False, sync_dist=True)

        for i in range(len(self.test_model_f1)):
            self.test_model_f1[i](preds=outputs['logits_model'], target=outputs['target_model'])
            self.log(f'test/model/f1' if i == 0 else f'test/model/f1_top{i+1}', self.test_model_f1[i],
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        self.test_precision(preds=outputs['logits_model'], target=outputs['target_model'])
        self.log(f'test/model/precision', self.test_precision,
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        self.test_recall(preds=outputs['logits_model'], target=outputs['target_model'])
        self.log(f'test/model/recall', self.test_recall,
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)


    def _step(self, batch):
        logits_model = self(x=batch['x'])

        if len(logits_model.shape) == 3:
            # We have a crop dimension that we need to average over
            logits_model = torch.mean(logits_model, dim=1)

        loss_model = F.cross_entropy(logits_model, batch['y'])
        loss = loss_model

        loss_dict = {'loss/model': loss_model}

        outputs = {'logits_model': logits_model,
                   'loss': loss, 'batch': batch, 'loss_dict': loss_dict,
                   'target_model': batch['y'],
                   }

        return outputs

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self._step(batch)

        self.log_training(outputs)

        return outputs['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self._step(batch)

        self.log_validation(outputs)

        return outputs['loss']

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self._step(batch)

        self.log_test(outputs)

        return outputs['loss']

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch)

    def forward(self, x):
        z = self.feature_extractor(x)

        logits_model = self.printer_model_out_layer(z)

        return logits_model
