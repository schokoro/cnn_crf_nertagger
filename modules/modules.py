from pdb import set_trace
from typing import Union, List, Dict, Tuple, Optional
from torch import nn
from torch.utils.data import TensorDataset
from allennlp.modules.conditional_random_field import ConditionalRandomField
import torch
from youtokentome import BPE

from utils.pipeline import predict_with_model
from utils.prepare import tokenize_corpus
import numpy as np


class StackedConv1d(nn.Module):
    """
    Базовый свёрточный модуль для NERTaggerModel
    """

    def __init__(self, features_num, layers_n=1,
                 kernel_size: Union[List[int], int] = 3,
                 dropout: float = 0.0,
                 dilation: Optional[List[int]] = None):
        """

        :param features_num:
        :param layers_n: int количество свёрточных слоёв
        :param kernel_size:
        :param dropout:
        :param dilation:
        """
        super().__init__()
        layers = []
        for i, _ in enumerate(range(layers_n)):
            if dilation is None:  # если None, то для всех слоёв dilation равен 1
                l_dilation = 1
            elif isinstance(dilation, list):
                l_dilation = dilation[i]
                assert len(dilation) == layers_n  # размер списка должен быть равен числу слоёв
            else:
                raise TypeError

            if isinstance(kernel_size, int):  # если int, то для всех слоёв размер свёртки равен kernel_size
                l_kernel_size = kernel_size
            elif isinstance(kernel_size, list):
                assert len(kernel_size) == layers_n  # размер списка должен быть равен числу слоёв
                l_kernel_size = kernel_size[i]
            else:
                raise TypeError

            layers.append(nn.Sequential(
                nn.Conv1d(
                    features_num, features_num, l_kernel_size,
                    padding=(l_kernel_size - 1) * l_dilation // 2, dilation=l_dilation
                ),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                # nn.BatchNorm1d(features_num)
            ))
            self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        :param x: - BatchSize x FeaturesNum x SequenceLen
        :return:
        """
        for layer in self.layers:
            x = x + layer(x)
        return x




class CNN_RNN_CRF(nn.Module):
    def __init__(self, vocab_size, labels_num, tag2id, embedding_size=32, single_backbone_kwargs={},
                 rnn_hidden_size=128, rnn_layer=2, dropout=.5):
        super().__init__()

        self.embedding_size = embedding_size
        self.char_embeddings = nn.Embedding(vocab_size+1, embedding_size, padding_idx=0)
        self.single_token_backbone = StackedConv1d(embedding_size, **single_backbone_kwargs)
        self.bn1 = nn.BatchNorm1d(self.embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.context_backbone = nn.GRU(embedding_size, rnn_hidden_size, num_layers=rnn_layer, bidirectional=True)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Linear(rnn_hidden_size * 2, labels_num)
        self.labels_num = labels_num
        STATE_TRANSITIONS_CONSTRAINTS = get_state_transitions_constraints(tag2id)
        self.crf = ConditionalRandomField(len(tag2id), constraints=STATE_TRANSITIONS_CONSTRAINTS)

    def forward(self, tokens):
        """

        :param tokens: BatchSize x MaxSentenceLen x MaxTokenLen
        :return:
        """
        batch_size, max_sent_len, max_token_len = tokens.shape
        tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)

        char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
        char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen
        char_embeddings = self.dropout(char_embeddings)
        char_features = self.single_token_backbone(char_embeddings)
        token_features_flat = self.global_pooling(char_features).squeeze(-1)  # BatchSize*MaxSentenceLen x EmbSize

        token_features = token_features_flat.view(batch_size, max_sent_len,
                                                  self.embedding_size)  # BatchSize x MaxSentenceLen x EmbSize
        context_features, _ = self.context_backbone(token_features)  # BatchSize x EmbSize x MaxSentenceLen
        logits = self.out(context_features)  # BatchSize x LabelsNum x MaxSentenceLen

        return logits.permute(0, 2, 1)


class CNN_CNN_CRF(nn.Module):
    def __init__(self,
                 vocab_size,
                 labels_num,
                 tag2id,
                 embedding_size=32,
                 single_backbone_kwargs=None,
                 context_backbone_kwargs=None, dropout1=0, dropout2=0):
        super().__init__()
        if context_backbone_kwargs is None:
            context_backbone_kwargs = {}
        self.embedding_size = embedding_size
        self.char_embeddings = nn.Embedding(vocab_size+1, embedding_size, padding_idx=0)
        self.single_token_backbone = StackedConv1d(embedding_size, **single_backbone_kwargs)
        self.bn1 = nn.BatchNorm1d(self.embedding_size)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.context_backbone = StackedConv1d(embedding_size, **context_backbone_kwargs)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Conv1d(embedding_size, labels_num, 1)
        self.labels_num = labels_num
        STATE_TRANSITIONS_CONSTRAINTS = get_state_transitions_constraints(tag2id)
        self.crf = ConditionalRandomField(len(tag2id), constraints=STATE_TRANSITIONS_CONSTRAINTS)

    def forward(self, tokens):
        """

        :param tokens: BatchSize x MaxSentenceLen x MaxTokenLen
        :return:
        """
        batch_size, max_sent_len, max_token_len = tokens.shape
        tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)

        char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
        char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen
        char_embeddings = self.dropout1(char_embeddings)
        char_features = self.single_token_backbone(char_embeddings)

        token_features_flat = self.global_pooling(char_features).squeeze(-1)  # BatchSize*MaxSentenceLen x EmbSize

        token_features = token_features_flat.view(batch_size, max_sent_len,
                                                  self.embedding_size)  # BatchSize x MaxSentenceLen x EmbSize

        token_features = token_features.permute(0, 2, 1)  # BatchSize x EmbSize x MaxSentenceLen
        token_features = self.bn1(token_features)
        token_features = self.dropout2(token_features)
        context_features = self.context_backbone(token_features)  # BatchSize x EmbSize x MaxSentenceLen
        logits = self.out(context_features)  # BatchSize x LabelsNum x MaxSentenceLen
        return logits


class NERTagger:
    """

    """

    def __init__(self, model: CNN_RNN_CRF, tokenizer: BPE, id2tag: dict, max_sent_len: int,
                 max_token_len: int, dropout: float = 0):
        """

        :param model:
        :param tokenizer:
        :param id2tag:
        :param max_sent_len:
        :param max_token_len:
        """
        self.model = model
        self.tokenizer = tokenizer
        self.id2tag = id2tag
        self.max_sent_len = max_sent_len
        self.max_token_len = max_token_len
        self.dropout = dropout

    def __call__(self, sentences):
        tokenized_corpus = tokenize_corpus(sentences)

        inputs = torch.zeros((len(sentences), self.max_sent_len, self.max_token_len + 2), dtype=torch.long)

        for sent_i, sentence in enumerate(tokenized_corpus):
            for token_i, token in enumerate(sentence):
                token_pieces = self.tokenizer.encode(token, dropout_prob=self.dropout)
                for piece_i, piece in enumerate(token_pieces):
                    inputs[sent_i, token_i, piece_i + 1] = piece

        dataset = TensorDataset(inputs, torch.zeros(len(sentences)))
        predicted_classes = predict_with_model(self.model, dataset).astype(np.int)

        result = []
        for sent_i, sent in enumerate(tokenized_corpus):
            result.append([self.id2tag[cls] for cls in predicted_classes[sent_i, :len(sent)]])
        return result


def get_state_transitions_constraints(tag2id: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Возвращает список допустимых переходов из тега в тег для CRF
    :param tag2id:
    :return:
    """
    str_transitions_constraints = [
        ('<NOTAG>', '<NOTAG>'), ('B-LOC', 'I-LOC'), ('B-MISC', 'I-MISC'), ('B-ORG', 'I-ORG'),
        ('I-LOC', '<NOTAG>'), ('I-LOC', 'B-LOC'), ('I-LOC', 'B-MISC'), ('I-LOC', 'B-ORG'), ('I-LOC', 'I-LOC'),
        ('I-LOC', 'I-MISC'), ('I-LOC', 'I-ORG'), ('I-LOC', 'I-PER'), ('I-LOC', 'O'),
        ('I-MISC', '<NOTAG>'), ('I-MISC', 'B-LOC'), ('I-MISC', 'B-MISC'), ('I-MISC', 'B-ORG'), ('I-MISC', 'I-LOC'),
        ('I-MISC', 'I-MISC'), ('I-MISC', 'I-ORG'), ('I-MISC', 'I-PER'), ('I-MISC', 'O'),
        ('I-ORG', '<NOTAG>'), ('I-ORG', 'B-LOC'), ('I-ORG', 'B-MISC'), ('I-ORG', 'B-ORG'), ('I-ORG', 'I-LOC'),
        ('I-ORG', 'I-MISC'), ('I-ORG', 'I-ORG'), ('I-ORG', 'I-PER'), ('I-ORG', 'O'),
        ('I-PER', '<NOTAG>'), ('I-PER', 'B-LOC'), ('I-PER', 'B-MISC'), ('I-PER', 'B-ORG'), ('I-PER', 'I-LOC'),
        ('I-PER', 'I-MISC'), ('I-PER', 'I-ORG'), ('I-PER', 'I-PER'), ('I-PER', 'O'),
        ('O', '<NOTAG>'), ('O', 'B-LOC'), ('O', 'B-MISC'), ('O', 'B-ORG'), ('O', 'I-LOC'),
        ('O', 'I-MISC'), ('O', 'I-ORG'), ('O', 'I-PER'), ('O', 'O')
    ]
    return [(tag2id[pair[0]], tag2id[pair[1]]) for pair in str_transitions_constraints]
