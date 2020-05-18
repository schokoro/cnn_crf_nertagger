import copy
import datetime
import traceback
import time
from typing import Dict, List, Tuple
from allennlp.modules.conditional_random_field import ConditionalRandomField
import torch
from torch.utils.data import DataLoader


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


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))


def train_eval_loop(model, train_dataset, val_dataset, tag2id,
                    lr=1e-4, epoch_n=10, batch_size=32,
                    device=None, early_stopping_patience=10, l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000,
                    max_batches_per_epoch_val=1000,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0,
                    verbose_batch=False):
    """
    Цикл для обучения модели. После каждой эпохи качество модели оценивается по отложенной выборке.
    :param model: torch.nn.Module - обучаемая модель
    :param train_dataset: torch.utils.data.Dataset - данные для обучения
    :param val_dataset: torch.utils.data.Dataset - данные для оценки качества
    :param criterion: функция потерь для настройки модели
    :param lr: скорость обучения
    :param epoch_n: максимальное количество эпох
    :param batch_size: количество примеров, обрабатываемых моделью за одну итерацию
    :param device: cuda/cpu - устройство, на котором выполнять вычисления
    :param early_stopping_patience: наибольшее количество эпох, в течение которых допускается
        отсутствие улучшения модели, чтобы обучение продолжалось.
    :param l2_reg_alpha: коэффициент L2-регуляризации
    :param max_batches_per_epoch_train: максимальное количество итераций на одну эпоху обучения
    :param max_batches_per_epoch_val: максимальное количество итераций на одну эпоху валидации
    :param data_loader_ctor: функция для создания объекта, преобразующего датасет в батчи
        (по умолчанию torch.utils.data.DataLoader)
    :param optimizer_ctor
    :param optimizer_params
    :param lr_scheduler_ctor
    :param shuffle_train
    :param dataloader_workers_n
    :return: кортеж из двух элементов:
        - среднее значение функции потерь на валидации на лучшей эпохе
        - лучшая модель
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters())


    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    STATE_TRANSITIONS_CONSTRAINTS = get_state_transitions_constraints(tag2id)
    crf = ConditionalRandomField(len(tag2id), constraints=STATE_TRANSITIONS_CONSTRAINTS)

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            print('Эпоха {}'.format(epoch_i))

            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):
                start_batch = time.time()
                if batch_i > max_batches_per_epoch_train:
                    break

                mask = (batch_x != tag2id['<NOTAG>'])

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)
                mask = copy_data_to_device(mask, device)

                pred = model(batch_x)
                loss = -crf(pred, batch_y, mask)
                # loss = criterion(pred, batch_y)

                model.zero_grad()
                loss.backward()

                optimizer.step()

                mean_train_loss += float(loss)
                train_batches_n += 1
                if verbose_batch:
                    print(f"Батч {batch_i} выполнен за {time.time() - start_batch:.2f} секунд")

            mean_train_loss /= train_batches_n
            print('Эпоха: {} итераций, {:0.2f} сек'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('Среднее значение функции потерь на обучении', mean_train_loss)

            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    mask = (batch_x != tag2id['<NOTAG>'])

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)
                    mask = copy_data_to_device(mask, device)

                    pred = model(batch_x)
                    loss = -crf(pred, batch_y, mask)

                    mean_val_loss += float(loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('Среднее значение функции потерь на валидации', mean_val_loss)

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('Новая лучшая модель!')
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                    early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
        except Exception as ex:
            print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
            break

    return best_val_loss, best_model


# decoded_tag_list_with_scores = crf.viterbi_tags(logits, mask)