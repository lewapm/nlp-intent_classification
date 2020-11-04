import torch
import codecs
import json
import numpy as np
import re
import nltk

from nltk.tokenize import word_tokenize
from torch.utils import data
nltk.download('punkt')


def remove_punctuation(sent):
    new_sent = []
    for word in sent:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_sent.append(new_word)
    return new_sent


def to_lowercase(sent):
    new_sent = []
    for word in sent:
        new_word = word.lower()
        new_sent.append(new_word)
    return new_sent


def normalize(sent):
    sent = word_tokenize(sent)
    sent = to_lowercase(sent)
    sent = remove_punctuation(sent)
    return sent


class Dataset(data.Dataset):
    def __init__(self, embedder, train=True, data_path='./data'):
        super(Dataset, self).__init__()
        self.embedder = embedder
        dir_names = ['BookRestaurant', 'SearchCreativeWork', 'PlayMusic', 'AddToPlaylist',
                     'RateBook', 'GetWeather', 'SearchScreeningEvent']
        file_name = 'train.json' if train else 'validation.json'
        self.sentences = []
        self.labels_dict = {}
        self.rev_labels_dict = {}
        for idx, name in enumerate(dir_names):
            self.labels_dict[idx] = name
            self.rev_labels_dict[name] = idx
            with codecs.open(data_path+"/"+name+"/"+file_name, 'r', encoding='utf-8',
                             errors='ignore') as json_file:
                load_sent = json.load(json_file)
                for sent in load_sent[name]:
                    cur_sent = ""
                    for words in sent['data']:
                        cur_sent += words['text']
                    cur_sent = self.embedder.normalize_and_add_words_from_sent_to_dict(cur_sent, train)
                    self.sentences.append((cur_sent, idx))

    def embed_sentences(self):
        new_sentences = []
        for sentence, label in self.sentences:
            embedding = self.embedder.create_sentence_embedding(sentence)
            new_sentences.append((embedding, sentence, label))
        self.sentences = new_sentences

    def get_labels_dict(self):
        return self.labels_dict

    def get_embedder(self):
        return self.embedder

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {'sentence': self.sentences[idx][0], 'length': len(self.sentences[idx][1]),
                'label': self.sentences[idx][2]}


class Embedder:
    """
    Class that creates words embeddings using Glove.6B.200d (Jeffrey Pennington, Richard Socher, and Christopher
    D. Manning. 2014. GloVe: Global Vectors for Word Representation.)
    """
    def __init__(self, load_path=None):
        if load_path is None:
            self.word_to_vec = torch.rand(3, 200)
            self.word_pos = {'xxbos': 0, 'xxeos': 1, 'xxunk': 2}
            self.words_saved = {}
            self.embed_len = 2
        else:
            self.load(load_path)

    def normalize_and_add_words_from_sent_to_dict(self, sent, train=False):
        sent = normalize(sent)
        self.embed_len = max(self.embed_len, len(sent) + 2)
        if train:
            for word in sent:
                if word not in self.words_saved:
                    self.words_saved[word] = len(self.words_saved)
        return sent

    def read_embeddings(self, path="./embedding/glove.6B.200d.txt"):
        """
        Read embeddings for words that are present in training dataset.
        """
        with open(path, 'r') as reader:
            for line in reader:
                line_split = line.strip().split(' ')
                if line_split[0] not in self.words_saved:
                    continue
                self.word_pos[line_split[0]] = len(self.word_pos)
                word_vec = torch.tensor(np.array([float(x) for x in line_split[1:]], dtype=np.float32))
                self.word_to_vec = torch.cat((self.word_to_vec, torch.unsqueeze(word_vec, 0)))

    def create_sentence_embedding(self, sent, predict=False):
        embed = torch.unsqueeze(self.word_to_vec[0], 0)
        for word in sent:
            if word not in self.word_pos:
                embed = torch.cat((embed, torch.unsqueeze(self.word_to_vec[2], 0)))
            else:
                embed = torch.cat((embed, torch.unsqueeze(self.word_to_vec[self.word_pos[word]], 0)))
        embed = torch.cat((embed, torch.unsqueeze(self.word_to_vec[1], 0)))
        if predict is False and embed.shape[0] < self.embed_len:
            embed = torch.cat((embed, torch.zeros(self.embed_len - embed.shape[0], 200)))
        return embed

    def save(self, path='./save'):
        self.word_pos['$$embed_len$$'] = self.embed_len
        torch.save(self.word_to_vec, path + '/word_to_vec.pt')
        with open(path + '/word_pos.json', 'w') as json_file:
            json.dump(self.word_pos, json_file)

    def load(self, path='./save'):
        with open(path + '/word_pos.json', 'r') as json_file:
            self.word_pos = json.load(json_file)
        self.word_to_vec = torch.load(path + '/word_to_vec.pt')
        self.embed_len = int(self.word_pos['$$embed_len$$'])