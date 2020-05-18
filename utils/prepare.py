from typing import Tuple

import torch
from tqdm import tqdm


def tag_corpus_to_tensor(sentences, char2id, tag2id, max_sent_len, max_token_len) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param sentences:
    :param char2id:
    :param tag2id:
    :param max_sent_len:
    :param max_token_len:
    :return inputs, targets: тензоры данных и таргет
    """
    # inputs [ len_corpus x max_sent_len x max_token_len + 2 ]
    # targets [ len_corpus x max_sent_len ]
    inputs = torch.zeros((len(sentences), max_sent_len, max_token_len + 2), dtype=torch.long)
    targets = torch.zeros((len(sentences), max_sent_len), dtype=torch.long)

    for sent_i, sent in tqdm(enumerate(sentences), total=len(sentences)):
        assert len(sent['tokens']) == len(sent['tags'])
        for token_i, token in enumerate(sent['tokens']):
            targets[sent_i, token_i] = tag2id[sent['tags'][token_i]]
            for char_i, char in enumerate(token.text):
                inputs[sent_i, token_i, char_i + 1] = char2id.get(char, 0)

    return inputs, targets




