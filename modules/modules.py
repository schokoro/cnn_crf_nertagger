from typing import Union, List
from torch import nn


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


class NERTagger(nn.Module):
    def __init__(self, vocab_size, labels_num, embedding_size=32, single_backbone_kwargs={},
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
        # set_trace()
        # nllh = self.crf(inputs=context_features.permute(0,2,1), tags=logits)
        return logits
