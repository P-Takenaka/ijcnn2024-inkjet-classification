import ml_collections
import os
import functools
import pickle

import torch.nn as nn
import torchmetrics

def get_config():
  config = ml_collections.ConfigDict()

  config.monitor_metric = 'val/model/f1'
  config.monitor_mode = 'max'

  config.data = ml_collections.ConfigDict({
        "data_dir": '.',
        "data_file": "extracted_features.pkl",
        "batch_size": 64,
  })

  with open(os.path.join(config.data.data_dir, config.data.data_file), 'rb') as f:
      data = pickle.load(f)

  config.model = ml_collections.ConfigDict({
        "module": "src.model.MLPModule",
        "input_size": data['metainfo']['num_features'],
        "hidden_sizes": [512, 512, 512],
        "activation_fn": nn.Tanh,
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 0.0001,
        },
  })

  return config
