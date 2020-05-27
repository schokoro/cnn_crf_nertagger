from typing import Tuple, Union, List, Dict
import youtokentome as yttm
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from allennlp.data.instance import Instance
import numpy as np
from ipymarkup import show_box_markup
# from ipymarkup.palette import palette, PALETTE, BLUE, RED, GREEN, PURPLE, BROWN, ORANGE, PURPLE


class ConllDataset(Dataset):

    def __init__(self, instances, tokenizer, tag2id, max_sent_len, max_token_len, augm=0, dropout=0):
        self.instances = instances
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_sent_len = max_sent_len
        self.max_token_len = max_token_len
        self.augm = augm
        self.dropout = dropout
        if augm > 0:




    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        self.instances[item]
        result = torch.zeros()


def make_yttm_tokenizer(train_conll: List[Instance], vocab_size=400):
    tokens = []
    for instance in train_conll:
        tokens += [token.text for token in instance['tokens']]
    text = ' '.join(tokens)

    with open('train_chunks.txt', 'w') as fobj:
        fobj.write(text)
    yttm.BPE.train(data='train_chunks.txt', vocab_size=vocab_size, model='conll_model.yttm')
    return yttm.BPE('conll_model.yttm')


def tag_corpus_to_tensor(sentences, tokenizer, tag2id, max_sent_len, max_token_len,
                         augm: Union[int, type(None)] = None, dropout=0) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param dropout:
    :param augm:
    :param sentences:
    :param tokenizer:
    :param tag2id:
    :param max_sent_len:
    :param max_token_len:
    :return inputs, targets: тензоры данных и таргет
    inputs [ len_corpus x max_sent_len x max_token_len + 2 ]
    targets [ len_corpus x max_sent_len ]
    """

    inputs = torch.zeros((len(sentences), max_sent_len, max_token_len + 2), dtype=torch.long)
    targets = torch.zeros((len(sentences), max_sent_len), dtype=torch.long)

    for sent_i, sent in tqdm(enumerate(sentences), total=len(sentences)):
        assert len(sent['tokens']) == len(sent['tags'])
        for token_i, token in enumerate(sent['tokens']):
            targets[sent_i, token_i] = tag2id[sent['tags'][token_i]]
            token_pieces = tokenizer.encode(token.text, dropout_prob=dropout)
            for piece_i, piece in enumerate(token_pieces):
                inputs[sent_i, token_i, piece_i + 1] = piece

    if augm:
        targ_idx = torch.zeros(targets.shape)
        for tag in ['B-LOC', 'B-MISC', 'B-ORG']:
            targ_idx += targets == tag2id[tag]
        targ_idx = targ_idx.int().sum(dim=1)
        augm_target = targets[targ_idx > 0]
        augm_target = augm_target.repeat(augm, 1)
        targets = torch.cat([targets, augm_target], dim=0)

        augm_inputs = inputs[targ_idx > 0]
        augm_inputs = augm_inputs.repeat(augm, 1, 1)
        inputs = torch.cat([inputs, augm_inputs], dim=0)

    return inputs, targets


def tokenize_corpus(texts):
    nlp = spacy.load('en_core_web_sm')
    return [[token.text for token in nlp.tokenizer(text)] for text in texts]


def tensor_to_tags(tens: Union[torch.Tensor, np.ndarray], id2tag:Dict[int, str]) -> List[List[str]]:
    """
    Преобразует тензор с айдишниками тегов в список списков тегов
    :param tens: входной тензор
    :param id2tag: словарь id -> тег
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
            if tens[i, j] != 0:
                sent_labels.append(id2tag[tens[i, j]])
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
