import os
import sys
import pathlib
import copy
import logging

from absl import app
from absl import flags

from dotenv import load_dotenv
from ml_collections import config_flags

import pytorch_lightning as pl

from src.dataset import ICTDataModule
from src.utils import get_module, init_module

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Config file.")
flags.DEFINE_integer('random_seed', 41, "Random Seed")

flags.mark_flags_as_required(["config"])

def init_logger():
    log_dir = f'logs'

    if os.path.exists(log_dir):
        index = 1
        adjusted_log_dir = f'{log_dir}_{index}'
        while os.path.exists(adjusted_log_dir):
            index += 1
            adjusted_log_dir = f'{log_dir}_{index}'

        log_dir = adjusted_log_dir

    os.makedirs(log_dir, exist_ok=False)

    # Logging Configuration
    formatter = logging.Formatter(
      fmt='%(asctime)s [%(levelname)s] : %(message)s',
      datefmt='%m/%d/%Y %H:%M:%S')

    logging.getLogger().setLevel(logging.INFO)

    logging.getLogger('pytorch_lightning').setLevel(logging.INFO)

    if log_dir:
        fh = logging.FileHandler(os.path.join(log_dir, 'output.log'), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)
        logging.getLogger('pytorch_lightning').addHandler(fh)

    return log_dir

def train_nn(config, config_path, log_dir  ):
    pl.seed_everything(config.seed)

    # Setup data
    data_module = ICTDataModule(**config.data.to_dict(), random_seed=config.seed)

    # Setup model
    model = init_module(
        config.model, num_printer_model_classes=data_module.num_printer_model_classes)

    callbacks = []

    monitor_metric = config.monitor_metric
    monitor_mode = config.monitor_mode

    callbacks.append(pl.callbacks.EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=10))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=monitor_metric,
            mode=monitor_mode, save_top_k=1)
    callbacks.append(checkpoint_callback)

    callbacks.append(pl.callbacks.ModelSummary(max_depth=50))

    callbacks.append(pl.callbacks.RichProgressBar())

    trainer = pl.Trainer(
        callbacks=callbacks, accelerator="cpu",
        devices=None,
        enable_progress_bar=True,
        max_epochs=99999, num_nodes=1)

    trainer.fit(model, data_module)

    model = get_module(config.model.module).load_from_checkpoint(
         checkpoint_callback.best_model_path, strict=True)

    # Evaluate on whole scans
    print("Testing the model")
    test_metrics = trainer.test(model, data_module)[0]

def main(argv):
    del argv
    config_param = ''
    for v in sys.argv:
        if v.startswith('--config'):
            config_param = v.split('=')[1]

    if not config_param:
        raise ValueError("cfg file not specified or in invalid format. it needs to be --config=CONFIG")

    config_path = os.path.join(os.getcwd(), config_param)

    config = FLAGS.config
    with config.unlocked():
        config.seed = FLAGS.random_seed

    log_dir = init_logger()
    if not log_dir:
        # Not rank zero, does not matter
        log_dir = '.'

    train_nn(config=config, config_path=config_path, log_dir=log_dir)

if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.resolve())
    load_dotenv(pathlib.Path(__file__).parent.joinpath('.env').resolve())

    app.run(main)
