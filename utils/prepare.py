from typing import Tuple, Union, List, Dict
import youtokentome as yttm
import spacy
import torch
from allennlp.data import Vocabulary
from torch.utils.data import Dataset
from tqdm import tqdm
from allennlp.data.instance import Instance
import numpy as np
from ipymarkup import show_box_markup
from youtokentome import BPE


class ConllDataset(Dataset):

    def __init__(self, instances: List[Instance], tokenizer: BPE, vocab: Vocabulary,
                 max_sent_len: int, max_token_len: int, augm: int = 0, dropout: float = 0):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_sent_len = max_sent_len
        self.max_token_len = max_token_len
        self.augm = augm
        self.dropout = dropout
        if augm > 0:
            add_instances = []
            b_tags = {'B-MISC', 'B-LOC', 'B-ORG'}
            for instance in instances:
                tags = set(instance['tags'])
                if tags.intersection(b_tags):
                    add_instances += [instance] * augm
            instances += add_instances
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        sent = self.instances[item]
        inputs = torch.zeros((self.max_sent_len, self.max_token_len + 2), dtype=torch.long)  # [max_sent_len x max_token_len + 2]
        targets = torch.ones(self.max_sent_len, dtype=torch.long) * self.vocab.get_token_index('[PAD]', 'labels') # [max_sent_len]
        assert len(sent['tokens']) == len(sent['tags'])
        for token_i, token in enumerate(sent['tokens']):
            targets[token_i] = self.vocab.get_token_index(sent['tags'][token_i], 'labels')
            token_pieces = self.tokenizer.encode(token.text, dropout_prob=self.dropout)
            for piece_i, piece in enumerate(token_pieces):
                inputs[token_i, piece_i + 1] = piece
        return inputs, targets


def make_yttm_tokenizer(train_conll: List[Instance], vocab_size=400):
    tokens = []
    for instance in train_conll:
        tokens += [token.text for token in instance['tokens']]
    text = ' '.join(tokens)

    with open('train_chunks.txt', 'w') as fobj:
        fobj.write(text)
    yttm.BPE.train(data='train_chunks.txt', vocab_size=vocab_size, model='conll_model.yttm')
    return yttm.BPE('conll_model.yttm')


def tokenize_corpus(texts):
    nlp = spacy.load('en_core_web_sm')
    return [[token.text for token in nlp.tokenizer(text)] for text in texts]


def tensor_to_tags(tens: Union[torch.Tensor, np.ndarray], vocab: Vocabulary) -> List[List[str]]:
    """
    Преобразует тензор с айдишниками тегов в список списков тегов
    :param vocab:
    :param tens: входной тензор
    :return: список со списками тегов для каждого предложения
    [['O', 'O', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['B-PER', 'I-PER'],
    ['I-LOC', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O']]
    """
    if type(tens) is torch.Tensor:
        tens = tens.numpy()
    n_sents, n_tokens = tens.shape
    labels = []
    for i in range(n_sents):
        sent_labels = []
        for j in range(n_tokens):
            if tens[i, j] != vocab.get_token_index('[PAD]', 'labels'):
                sent_labels.append(vocab.get_token_from_index(tens[i, j], 'labels'))
        labels.append(sent_labels)
    return labels


def highlight_text(tokens: List[str], tags: List[str]):
    """
    Выводит на печать текст с подсветкой тегов
    :param tokens: список токенов
    :param tags: список тегов
    :return:
    """
    assert len(tokens) == len(tags)
    spans = []
    start = 0
    for i in range(len(tokens)):
        len_token = len(tokens[i])
        spans.append((start, start + len_token, tags[i]))
        start += len_token + 1
    text = ' '.join(tokens)
    show_box_markup(text, spans)
