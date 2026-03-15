"""
Восстановление обозначений классов: повреждённые строки -> целые 0, 1, 2, ...
"""
import numpy as np


def get_label_mapping(labels: np.ndarray) -> dict:
    """
    Строит взаимно однозначное отображение уникальных значений (строк) в целые 0..n-1.
    Порядок — по первой встрече в массиве (или отсортированный для детерминизма).
    """
    flat = np.asarray(labels).ravel()
    # Уникальные значения в порядке появления (или сортировка для стабильности)
    unique = np.unique(flat.astype(str))
    return {str(u): i for i, u in enumerate(unique)}


def restore_labels(labels: np.ndarray, mapping: dict = None):
    """
    Преобразует метки в целые. Если mapping не передан — строится по данным.
    Возвращает (целочисленный массив меток, mapping).
    """
    labels = np.asarray(labels)
    if mapping is None:
        mapping = get_label_mapping(labels)
    restored = np.array([mapping[str(l)] for l in labels.ravel()], dtype=np.int32)
    if labels.shape != restored.shape:
        restored = restored.reshape(labels.shape)
    return restored, mapping


def restore_labels_from_train_valid(train_y: np.ndarray, valid_y: np.ndarray) -> tuple:
    """
    Единый mapping по train_y и valid_y, чтобы классы совпадали.
    Возвращает (train_y_restored, valid_y_restored, mapping).
    """
    all_labels = np.concatenate([np.ravel(train_y), np.ravel(valid_y)])
    mapping = get_label_mapping(all_labels)
    train_restored, _ = restore_labels(train_y, mapping)
    valid_restored, _ = restore_labels(valid_y, mapping)
    return train_restored, valid_restored, mapping
