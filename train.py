import torch
import numpy as np

from dataset import Dataset, Embedder
from model import IntentRecognizer
from pathlib import Path

params={
    "lr": 1e-3,
    "optimizer": "Adam",
    "epochs": 3,
    "labels": 7,
    "batch_size": 32,
    "sizes": [200, 100],
    'model_save_dir': './trained_model'
}

Path(params['model_save_dir']).mkdir(parents=True, exist_ok=True)
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
embedder = Embedder()
train_dataset = Dataset(embedder=embedder)
valid_dataset = Dataset(embedder=embedder, train=False)
embedder.read_embeddings()
train_dataset.embed_sentences()
valid_dataset.embed_sentences()
recognizer = IntentRecognizer()
recognizer.train(train_dataset, valid_dataset, params)
