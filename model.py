import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import tqdm

from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score
from dataset import Dataset, normalize, Embedder


class LSTMRecognizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, tagset_size):
        super(LSTMRecognizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.linear1 = nn.Linear(hidden_dim, 200)
        self.linear2 = nn.Linear(200, tagset_size)
        self.hidden = self.reset_hidden()

    def reset_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                       torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, sent, sent_len):
        self.hidden = self.reset_hidden()
        X = torch.nn.utils.rnn.pack_padded_sequence(sent, sent_len, batch_first=True, enforce_sorted=False)
        _, self.hidden = self.lstm(X, self.hidden)
        x = self.linear1(self.hidden[0])
        x = F.relu(x)
        x = self.linear2(x)
        tag_scores = F.softmax(x, dim=2)
        return tag_scores


class IntentRecognizer:
    def __init__(self, embedder=None):
        self.model = None
        self.embedder = embedder

    def prepare_model(self, training_params):
        self.model = LSTMRecognizer(training_params['sizes'][0], training_params['sizes'][1],
                                    training_params['batch_size'], training_params['labels'])

    def train_epoch(self, train_loader, batch_size):
        loss_fn = nn.CrossEntropyLoss()
        y_true = []
        y_pred = []
        for batch_idx, sample in enumerate(tqdm.tqdm(train_loader)):
            self.optimizer.zero_grad()
            X = sample['sentence']
            len = sample['length']
            y = sample['label']
            preds = self.model.forward(X, len)
            preds = preds.view(batch_size, -1)
            loss = loss_fn(preds, y)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            y_pred += torch.argmax(preds, dim=1).tolist()
            y_true += y.tolist()
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        return f1, acc

    def eval_model(self, valid_loader, batch_size):
        y_true = []
        y_pred = []
        for batch_idx, sample in enumerate(valid_loader):
            X = sample['sentence']
            len = sample['length']
            y = sample['label']
            preds = self.model.forward(X, len)
            preds = preds.view(batch_size, -1)
            y_pred += torch.argmax(preds, dim=1).tolist()
            y_true += y.tolist()
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        return f1, acc

    def save_results_to_json(self, path, train_f1, train_acc, valid_f1, valid_acc):
        results = {
            'train': {
                'accuracy': train_acc,
                'f1': train_f1
            },
            'validation': {
                'accuracy': valid_acc,
                'f1': valid_f1
            }
        }
        with open(path+'/metrics.json', 'w') as json_file:
            json.dump(results, json_file)

    def save_model(self, path='./saved'):
        filename = path+"/weights.ckpt"
        torch.save(self.model, filename)

    def save_labels_dict(self, path='./saved'):
        with open(path+'/labels_dict.json', 'w') as json_file:
            json.dump(self.labels_dict, json_file)

    def load_saved_model(self, path='./saved'):
        self.model = torch.load(path+'/weights.ckpt')

    def load_labels_dict(self, path='./saved'):
        self.labels_dict = {}
        with open(path + '/labels_dict.json', 'r') as json_file:
            load_dict = json.load(json_file)
            for key, value in load_dict.items():
                self.labels_dict[int(key)] = value

    def train(self, train_dataset: Dataset, validation_dataset: Dataset, training_params: Optional[dict]) -> None:
        print(training_params)
        train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, drop_last=True)
        valid_loader = DataLoader(validation_dataset, batch_size=training_params['batch_size'], shuffle=True,
                                  drop_last=True)
        if self.model is None:
            self.prepare_model(training_params)
            self.embedder = train_dataset.get_embedder()
            self.labels_dict = train_dataset.get_labels_dict()

        if training_params['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=training_params["lr"])
        elif training_params['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=training_params["lr"])

        train_f1, train_acc, valid_f1, valid_acc = 0, 0, 0, 0
        best_acc = 0
        for epoch in range(training_params['epochs']):
            train_f1, train_acc = self.train_epoch(train_loader, training_params['batch_size'])
            print(f"After epoch {epoch} for train acc: {train_acc:.3f}  f1: {train_f1:.3f}")
            valid_f1, valid_acc = self.eval_model(valid_loader, training_params['batch_size'])
            print(f"After epoch {epoch} for valid acc: {valid_acc:.3f}  f1: {valid_f1:.3f}")
            if valid_acc > best_acc:
                self.save_results_to_json(training_params['model_save_dir'], train_f1, train_acc, valid_f1, valid_acc)
                self.save_model(training_params['model_save_dir'])

        self.save_labels_dict(training_params['model_save_dir'])
        self.embedder.save(training_params['model_save_dir'])

    def predict(self, utterance: str) -> dict:
        if self.embedder is None or self.model is None:
            raise ('Load or train model, before predict')
        sent = normalize(utterance)
        sent = self.embedder.create_sentence_embedding(sent, predict=True)
        sent = torch.unsqueeze(sent, 0)
        with torch.no_grad():
            x = self.model.forward(sent, torch.tensor([sent.shape[1]])).view(-1)
            predicted = torch.argmax(x).tolist()
            return {'intent': self.labels_dict[predicted], 'confidence': x[predicted].tolist()}

    def load_model(self, model_dirpath: str) -> None:
        self.load_saved_model(model_dirpath)
        self.load_labels_dict(model_dirpath)
        self.embedder = Embedder(model_dirpath)