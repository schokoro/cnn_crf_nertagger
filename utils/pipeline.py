import copy
import datetime
import traceback
import time
from typing import Dict, List, Tuple, Union
from livelossplot import PlotLosses
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from pdb import set_trace


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))


def train_eval_loop(model: Module, train_dataset: Dataset, val_dataset: Dataset,
                    lr: float = 1e-4, epoch_n: int = 10, batch_size: int = 32,
                    device=None, early_stopping_patience: int = 10, l2_reg_alpha: float = 0,
                    max_batches_per_epoch_train: int = 10000,
                    max_batches_per_epoch_val: int = 1000,
                    optimizer_ctor: Optimizer = None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n: int = 0,
                    verbose_batch: bool = False,
                    verbose_liveloss=True,
                    prev_loss: Dict[str, List[float]] = {}) -> Tuple[float, Module, Dict[str, List[float]]]:
    """
    Цикл для обучения модели. После каждой эпохи качество модели оценивается по отложенной выборке.
    :param prev_loss: лоссы от предыдущего цикла обучения
    :param verbose_batch:
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=dataloader_workers_n)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    losses = {'loss': prev_loss.get('train_loss', []), 'val_loss': prev_loss.get('val_loss', [])}
    if verbose_liveloss:
        liveloss = PlotLosses()
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

                mask = (batch_x[:, :, 1] != 0)

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)
                mask = copy_data_to_device(mask, device)
                # set_trace()
                pred = model(batch_x)

                loss = -model.crf(pred.permute(0, 2, 1), batch_y, mask) / batch_size
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
            losses['loss'].append(mean_train_loss)

            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    mask = (batch_x[:, :, 1] != 0)

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)
                    mask = copy_data_to_device(mask, device)

                    pred = model(batch_x)
                    loss = -model.crf(pred.permute(0, 2, 1), batch_y, mask) / batch_size

                    mean_val_loss += float(loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('Среднее значение функции потерь на валидации', mean_val_loss)
            losses['val_loss'].append(mean_val_loss)

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
        if verbose_liveloss:
            liveloss.update(losses)
            liveloss.send()

    return best_val_loss, best_model, losses


def paths_to_tensor(batch_pred_paths, size) -> torch.Tensor:
    """

    :param batch_pred_paths:
    :param size:
    :return:
    """
    batch_pred = torch.zeros(size)
    for i, bp in enumerate(batch_pred_paths):
        batch_pred[i, : len(bp[0])] = torch.Tensor(bp[0])
    return batch_pred


def predict_with_model(model, dataset, device=None, batch_size=32, num_workers=0, return_labels=False):
    """
    :param model: torch.nn.Module - обученная модель
    :param dataset: torch.utils.data.Dataset - данные для применения модели
    :param device: cuda/cpu - устройство, на котором выполнять вычисления
    :param batch_size: количество примеров, обрабатываемых моделью за одну итерацию
    :return: numpy.array размерности len(dataset) x *
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    with torch.no_grad():
        import tqdm
        for batch_x, batch_y in tqdm.tqdm(dataloader, total=len(dataset) / batch_size):
            batch_x = copy_data_to_device(batch_x, device)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred = model(batch_x)
            mask = (batch_x[:, :, 1] != 0)
            batch_pred = model.crf.viterbi_tags(batch_pred.permute(0, 2, 1), mask)
            batch_pred = paths_to_tensor(batch_pred, mask.shape)
            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)
