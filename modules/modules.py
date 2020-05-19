from typing import Union, List, Dict, Tuple
from torch import nn
from torch.utils.data import TensorDataset
from allennlp.modules.conditional_random_field import ConditionalRandomField
import torch

from utils.pipeline import predict_with_model


class StackedConv1d(nn.Module):
    def __init__(self, features_num, layers_n=1,
                 kernel_size: Union[List[int], int] = 3,
                 conv_layer: nn.Module = nn.Conv1d,
                 dropout: float = 0.0,
                 dilation: Union[List[int], type(None)] = None):
        """

        :param features_num:
        :param layers_n:
        :param kernel_size:
        :param conv_layer:
        :param dropout:
        :param dilation:
        """
        super().__init__()
        layers = []
        for i, _ in enumerate(range(layers_n)):
            if dilation is None:  # если None, то для всех слоёв dilation равен 1
                l_dilation = 1
            elif type(dilation) == list:
                l_dilation = dilation[i]
                assert len(dilation) == layers_n  # размер списка должен быть равен числу слоёв
            else:
                raise TypeError

            if type(kernel_size) == int:  # если int, то для всех слоёв размер свёртки равен kernel_size
                l_kernel_size = kernel_size
            elif type(kernel_size) == list:
                assert len(kernel_size) == layers_n  # размер списка должен быть равен числу слоёв
                l_kernel_size = kernel_size[i]
            else:
                raise TypeError

            layers.append(nn.Sequential(
                conv_layer(
                    features_num, features_num, l_kernel_size,
                    padding=(l_kernel_size - 1) * l_dilation // 2, dilation=l_dilation
                ),
                nn.Dropout(dropout),
                nn.LeakyReLU()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        :param x: - BatchSize x FeaturesNum x SequenceLen
        :return:
        """
        for layer in self.layers:
            x = x + layer(x)
        return x


def get_state_transitions_constraints(tag2id: Dict[str, int]) -> List[Tuple[int, int]]:
    """

    :param tag2id:
    :return:
    """
    str_transitions_constraints = [
        ('<NOTAG>', '<NOTAG>'), ('B-LOC', 'I-LOC'), ('B-MISC', 'I-MISC'), ('B-ORG', 'I-ORG'),
        ('I-LOC', '<NOTAG>'), ('I-LOC', 'B-LOC'), ('I-LOC', 'B-MISC'), ('I-LOC', 'B-ORG'),
        ('I-LOC', 'I-LOC'), ('I-LOC', 'I-MISC'), ('I-LOC', 'I-ORG'), ('I-LOC', 'I-PER'),
        ('I-LOC', 'O'), ('I-MISC', '<NOTAG>'), ('I-MISC', 'B-LOC'), ('I-MISC', 'B-MISC'),
        ('I-MISC', 'B-ORG'), ('I-MISC', 'I-LOC'), ('I-MISC', 'I-MISC'), ('I-MISC', 'I-ORG'),
        ('I-MISC', 'I-PER'), ('I-MISC', 'O'), ('I-ORG', '<NOTAG>'), ('I-ORG', 'B-LOC'),
        ('I-ORG', 'B-MISC'), ('I-ORG', 'B-ORG'), ('I-ORG', 'I-LOC'), ('I-ORG', 'I-MISC'),
        ('I-ORG', 'I-ORG'), ('I-ORG', 'I-PER'), ('I-ORG', 'O'), ('I-PER', '<NOTAG>'),
        ('I-PER', 'B-LOC'), ('I-PER', 'B-MISC'), ('I-PER', 'B-ORG'), ('I-PER', 'I-LOC'),
        ('I-PER', 'I-MISC'), ('I-PER', 'I-ORG'), ('I-PER', 'I-PER'), ('I-PER', 'O'),
        ('O', '<NOTAG>'), ('O', 'B-LOC'), ('O', 'B-MISC'), ('O', 'B-ORG'), ('O', 'I-LOC'),
        ('O', 'I-MISC'), ('O', 'I-ORG'), ('O', 'I-PER'), ('O', 'O')
    ]
    return [(tag2id[pair[0]], tag2id[pair[1]]) for pair in str_transitions_constraints]


class NERTaggerModel(nn.Module):
    def __init__(self, vocab_size, labels_num, tag2id, embedding_size=32, single_backbone_kwargs={},
                 context_backbone_kwargs=None):
        super().__init__()
        if context_backbone_kwargs is None:
            context_backbone_kwargs = {}
        self.embedding_size = embedding_size
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.single_token_backbone = StackedConv1d(embedding_size, **single_backbone_kwargs)
        self.context_backbone = StackedConv1d(embedding_size, **context_backbone_kwargs)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Conv1d(embedding_size, labels_num, 1)
        self.labels_num = labels_num
        STATE_TRANSITIONS_CONSTRAINTS = get_state_transitions_constraints(tag2id)
        self.crf = ConditionalRandomField(len(tag2id), constraints=STATE_TRANSITIONS_CONSTRAINTS)

    def forward(self, tokens):
        """tokens - BatchSize x MaxSentenceLen x MaxTokenLen"""
        batch_size, max_sent_len, max_token_len = tokens.shape
        tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)

        char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
        char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen
        char_features = self.single_token_backbone(char_embeddings)

        token_features_flat = self.global_pooling(char_features).squeeze(-1)  # BatchSize*MaxSentenceLen x EmbSize

        token_features = token_features_flat.view(batch_size, max_sent_len,
                                                  self.embedding_size)  # BatchSize x MaxSentenceLen x EmbSize
        token_features = token_features.permute(0, 2, 1)  # BatchSize x EmbSize x MaxSentenceLen
        context_features = self.context_backbone(token_features)  # BatchSize x EmbSize x MaxSentenceLen
        logits = self.out(context_features)  # BatchSize x LabelsNum x MaxSentenceLen

        return logits


class POSTagger:
    def __init__(self, model, char2id, id2label, max_sent_len, max_token_len):
        self.model = model
        self.char2id = char2id
        self.id2label = id2label
        self.max_sent_len = max_sent_len
        self.max_token_len = max_token_len

    def __call__(self, sentences):
        tokenized_corpus = tokenize_corpus(sentences, min_token_size=1)

        inputs = torch.zeros((len(sentences), self.max_sent_len, self.max_token_len + 2), dtype=torch.long)

        for sent_i, sentence in enumerate(tokenized_corpus):
            for token_i, token in enumerate(sentence):
                for char_i, char in enumerate(token):
                    inputs[sent_i, token_i, char_i + 1] = self.char2id.get(char, 0)

        dataset = TensorDataset(inputs, torch.zeros(len(sentences)))
        predicted_probs = predict_with_model(self.model, dataset)  # SentenceN x TagsN x MaxSentLen
        predicted_classes = predicted_probs.argmax(1)

        result = []
        for sent_i, sent in enumerate(tokenized_corpus):
            result.append([self.id2label[cls] for cls in predicted_classes[sent_i, :len(sent)]])
        return result
