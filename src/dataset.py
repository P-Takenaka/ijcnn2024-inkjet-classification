import copy
import numpy as np
import pickle
import os

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torchvision.transforms as T
import pytorch_lightning as pl

class ICTDataset(torch.utils.data.Dataset):
    def __init__(self, df, crop_size, mean, std, take_crop):
        super().__init__()

        self.mean = mean
        self.std = std
        self.crop_size = crop_size
        self.take_crop = take_crop

        if self.crop_size:
            self.crop_transforms = T.Compose([T.ToTensor(), T.CenterCrop(self.crop_size)])
        else:
            self.crop_transforms = None

        self.df = df

        if type(df) == list:
            # Grouped by document
            self.is_grouped = True
        else:
            self.is_grouped = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.is_grouped:
            sample = self.df[idx]

            x = np.stack([v for v in sample['x']], axis=0).astype(np.float32) if len(sample) > 1 else sample['x'].astype(np.float32)
            if len(x.shape) == 2:
                y = np.array(sample['y'].iloc[0], dtype=np.int64)
            else:
                y = np.array(sample['y'], dtype=np.int64)
        else:
            sample = self.df.iloc[idx]

            x = np.array(sample['x'], dtype=np.float32)
            y = np.array(sample['y'], dtype=np.int64)

        x = (x - self.mean) / (self.std + 1e-4)

        x = np.where(self.std < 1e-4, 1.0, x)

        res = {'x': x, 'y': y,
               }

        if self.take_crop:
            if self.is_grouped:
                res['crop'] = torch.stack([self.crop_transforms(crop) for crop in sample['crop']], dim=0)
            else:
                assert(self.crop_transforms is not None)
                res['crop'] = self.crop_transforms(sample['crop'])

        return res


class ICTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, data_file, batch_size: int, random_seed, take_crop=False, num_segments=None, average_features=False, mean_document_samples=False, group_document_samples_during_train=False, group_document_samples_during_val=False):
        super().__init__()

        self.data_file = data_file
        self.img_base_folder = './images/output'
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.take_crop = take_crop
        self.average_features = average_features
        self.group_document_samples_during_train = group_document_samples_during_train
        self.group_document_samples_during_val = group_document_samples_during_val
        self.mean_document_samples = mean_document_samples

        self.train_batch_size = batch_size
        self.val_batch_size = batch_size

        if self.group_document_samples_during_train:
            self.train_batch_size = 1

        if self.group_document_samples_during_val:
            self.val_batch_size = 1

        with open(os.path.join(data_dir, self.data_file), 'rb') as f:
            self.data = pickle.load(f)

        self.num_segments = self.data['metainfo']['num_crops_per_image'] if num_segments is None else num_segments
        self.feature_crop_size = self.data['metainfo']['crop_size']

        self.num_printer_model_classes = self.data['metainfo']['num_classes']
        self.train_printer_models = list(self.data['metainfo']['printer_models'])

        self.train_df = self.data['train_df']

        if self.group_document_samples_during_train:
            train_df_rows = []
            # Group all crops belonging to the same document
            for filename in self.train_df['filename'].unique():
                rows = self.train_df[self.train_df['filename'] == filename]

                if self.mean_document_samples:
                    row = rows.iloc[0]
                    row['x'] = rows['x'].mean().astype(np.float32)
                    row['crop'] = None

                    train_df_rows.append(row)
                else:
                    train_df_rows.append(rows)

            self.train_df = train_df_rows

            # Normalization
            # Normalize each document first
            means = np.array([f['x'] for f in self.train_df]) if self.mean_document_samples else np.array([f['x'].mean() for f in self.train_df])

            self.train_feature_means = np.mean(means, axis=0).astype(np.float32)
            self.train_feature_stddev = np.std(means, axis=0).astype(np.float32)
        else:
            # For normalizations
            features = np.array(list(self.train_df['x']), dtype=np.float32)
            self.train_feature_means = np.mean(features, axis=0)
            self.train_feature_stddev = np.std(features, axis=0)


        self.val_df = self.data['val_df']

        val_df_rows = []
        # Group all crops belonging to the same document
        for filename in self.val_df['filename'].unique():
            rows = self.val_df[self.val_df['filename'] == filename]

            if self.mean_document_samples:
                row = rows.iloc[0]
                row['x'] = rows['x'].mean().astype(np.float32)
                row['crop'] = None

                val_df_rows.append(row)
            else:
                val_df_rows.append(rows)

        self.grouped_val_df = val_df_rows
        if self.group_document_samples_during_val:
            self.val_df = copy.deepcopy(self.grouped_val_df)


        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str):
        self.train_ds = ICTDataset(
            self.train_df, mean=self.train_feature_means,
            std=self.train_feature_stddev, crop_size=self.feature_crop_size,
            take_crop=self.take_crop)

        self.val_ds = ICTDataset(
            self.val_df, mean=self.train_feature_means, std=self.train_feature_stddev,
            crop_size=self.feature_crop_size, take_crop=self.take_crop)

        self.test_ds = ICTDataset(
            self.grouped_val_df, mean=self.train_feature_means, std=self.train_feature_stddev,
            crop_size=self.feature_crop_size, take_crop=self.take_crop)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

        return loader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=4)
