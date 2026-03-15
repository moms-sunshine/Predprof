import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ml.restore_labels import get_label_mapping, restore_labels, restore_labels_from_train_valid


def test_get_label_mapping():
    labels = np.array(['a', 'b', 'a', 'c'])
    m = get_label_mapping(labels)
    assert m == {'a': 0, 'b': 1, 'c': 2}


def test_restore_labels():
    labels = np.array(['x', 'y', 'x'])
    restored, m = restore_labels(labels)
    assert list(restored) == [0, 1, 0]
    assert m['x'] == 0 and m['y'] == 1


def test_restore_labels_from_train_valid():
    train_y = np.array(['c', 'a', 'b'])
    valid_y = np.array(['a', 'c'])
    t, v, m = restore_labels_from_train_valid(train_y, valid_y)
    assert list(t) == [2, 0, 1]
    assert list(v) == [0, 2]
    assert len(m) == 3
