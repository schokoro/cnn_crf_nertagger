
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
if device == 'cpu':
    print('cpu')
else:
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.common.util import ensure_list
from multiprocessing import cpu_count
from modules.modules import CNN_RNN_CRF, NERTagger, CNN_CNN_CRF
from utils.pipeline import train_eval_loop, predict_with_model
from utils.prepare import tag_corpus_to_tensor, tokenize_corpus, highlight_text, tensor_to_tags, ConllDataset
from os import path, listdir
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from seqeval.metrics import classification_report

torch.backends.cudnn.deterministic = False


data_path = 'data'
path_data = f'./data/'
path_train = f'{data_path}/eng.train'
path_valid = f'{data_path}/eng.testa'
path_test = f'{data_path}/eng.testb'

# dataset_urls = {
#     'eng.testa': 'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa',
#     'eng.testb': 'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb',
#     'eng.train': 'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train'}
# for file_name in dataset_urls:
#     wget.download(dataset_urls[file_name], path.join(path_data, file_name))



conll_reader = Conll2003DatasetReader()
train_conll = ensure_list(conll_reader.read(path_train))
valid_conll = ensure_list(conll_reader.read(path_valid))
test_conll = ensure_list(conll_reader.read(path_test))

all_conll = train_conll + valid_conll + test_conll
len(all_conll), len(train_conll), len(valid_conll), len(test_conll)

tags = set()
tokens = set()

max_sent_len = 0
for instance in all_conll[:]:
    if len(instance['tokens']) > max_sent_len:
        max_sent_len = len(instance['tokens'])
    tags.update(instance['tags'])
    tokens.update([t.text for t in instance['tokens']])

print(f'Максимальная длина предложения: {max_sent_len} токенов')

chars = set()
for token in tokens:
    chars.update(token)
tag2id = {tag: num for num, tag in enumerate(['<NOTAG>'] + list(tags))}
char2id = {char: num+1 for num, char in enumerate(chars)}

BPE_DROPOUT = .25
max_token_len = max([len(token) for token in tokens])
train_dataset = ConllDataset(train_conll, char2id, tag2id, max_sent_len, max_token_len)
valid_dataset = ConllDataset(valid_conll, char2id, tag2id, max_sent_len, max_token_len)
test_dataset = ConllDataset(test_conll, char2id, tag2id, max_sent_len, max_token_len)

len(train_dataset), len(valid_dataset), len(test_dataset) 

models_path = './models/best_model.pth'


torch.cuda.empty_cache()
# model = CNN_RNN_CRF(
#     len(char2id), len(tag2id), tag2id, embedding_size=64,
#     single_backbone_kwargs=dict(layers_n=3, kernel_size=3, dropout=0.2, dilation=[1, 1, 1]),
#     rnn_hidden_size=256, rnn_layer=2, dropout=0)

model = CNN_CNN_CRF(
    len(char2id), len(tag2id), tag2id, embedding_size=64,
    single_backbone_kwargs=dict(layers_n=3, kernel_size=3, dropout=0.2, dilation=[1, 1, 1]),
    context_backbone_kwargs=dict(layers_n=6, kernel_size=3, dropout=0.1, dilation=[1, 1, 1,  2, 2, 2]),  dropout1=0.3)

print('Количество параметров', sum(np.product(t.shape) for t in model.parameters()))

losses = {}


(best_val_loss,
 best_model,
 losses) = train_eval_loop(
    model,
    train_dataset,
    valid_dataset,
    lr=1e-3,
    epoch_n=200,
    batch_size=256,
    device=device,
    early_stopping_patience=8,
    l2_reg_alpha=1e-6,
    max_batches_per_epoch_train=50,
    max_batches_per_epoch_val=50,
    dataloader_workers_n=cpu_count(),
    # optimizer_ctor=lambda params: torch.optim.SGD(
    #     params,
    #     lr=1e-2,
    #     weight_decay=1e-6
    # ),
    lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=4,
        factor=0.1,
        threshold=1e-2,
        verbose=True,
        min_lr=1e-5),
    verbose_batch=False,
    verbose_liveloss=False,
    prev_loss=losses
)
 
torch.save(best_model.state_dict(), models_path)

pd.DataFrame(losses).plot()

model.load_state_dict(torch.load(models_path))

id2tag = {item[1]: item[0] for item in tag2id.items()}
UNIQUE_TAGS = [id2tag[i] for i in range(len(tag2id))]

train_targets = [item[1] for item in train_dataset]
train_targets = torch.stack(train_targets)
print(train_targets.shape)

train_pred = predict_with_model(model, train_dataset)
train_golden_tags = tensor_to_tags(train_targets, id2tag)
train_pred_tags = tensor_to_tags(train_pred, id2tag)
print(classification_report(train_golden_tags, train_pred_tags, digits=4))
print(classification_report(train_golden_tags, train_pred_tags, digits=4, suffix=True))


# ### Проверка - valid
valid_targets = [item[1] for item in valid_dataset]
valid_targets = torch.stack(valid_targets)
valid_pred = predict_with_model(model, valid_dataset)

 
valid_golden_tags = tensor_to_tags(valid_targets, id2tag)
valid_pred_tags = tensor_to_tags(valid_pred, id2tag)
print(classification_report(valid_golden_tags, valid_pred_tags, digits=4))
print(classification_report(valid_golden_tags, valid_pred_tags, digits=4, suffix=True))


# ### Проверка - test

test_targets = [item[1] for item in test_dataset]
test_targets = torch.stack(test_targets)
print(f"test_targets.shape = {test_targets.shape}")


test_pred = predict_with_model(model, test_dataset)
 
test_golden_tags = tensor_to_tags(test_targets, id2tag)
test_pred_tags = tensor_to_tags(test_pred, id2tag)
print(classification_report(test_golden_tags, test_pred_tags, digits=4))
print(classification_report(test_golden_tags, test_pred_tags, digits=4, suffix=True))
