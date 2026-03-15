"""
Скрипт для обучения модели в Яндекс DataSphere.
Скачай Data.npz с https://disk.yandex.ru/d/BA4oJb0BwaABxg и укажи путь к нему в DATA_PATH.
После обучения сохраняются: model.h5, training_history.json, train_class_counts.json, valid_top5.json.
Запуск из корня проекта: python ml/train_dataspere.py
"""
import json
import numpy as np
import os
import sys

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Чтобы импорт ml работал при запуске из любой папки
sys.path.insert(0, os.path.dirname(OUT_DIR))

DATA_PATH = os.path.join(os.path.dirname(OUT_DIR), 'Data.npz')

def main():
    from ml.restore_labels import restore_labels_from_train_valid

    data = np.load(DATA_PATH, allow_pickle=True)
    train_x = data['train_x']
    train_y = data['train_y']
    valid_x = data['valid_x']
    valid_y = data['valid_y']

    train_y, valid_y, label_mapping = restore_labels_from_train_valid(train_y, valid_y)
    n_classes = len(label_mapping)

    # Приводим сигналы к одному размеру: каждый элемент train_x — массив (длина может быть разная)
    def to_matrix(arr):
        arr = np.array([np.array(x, dtype=np.float32).ravel() for x in arr])
        max_len = max(len(x) for x in arr)
        out = np.zeros((len(arr), max_len), dtype=np.float32)
        for i, x in enumerate(arr):
            out[i, :len(x)] = x
        return out

    X_train = to_matrix(train_x)
    X_valid = to_matrix(valid_x)

    import tensorflow as tf
    from tensorflow import keras

    # Простая сеть: несколько плотных слоёв
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    y_train_cat = train_y
    y_valid_cat = valid_y

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_valid, y_valid_cat),
        epochs=30,
        batch_size=32,
        verbose=1,
    )

    # Сохраняем модель
    model.save(os.path.join(OUT_DIR, 'model.h5'))

    # История для графика "точность от эпох"
    hist = {
        'epochs': list(range(1, len(history.history['val_accuracy']) + 1)),
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    }
    with open(os.path.join(OUT_DIR, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(hist, f, indent=2)

    # Количество записей по классам в обучении
    unique, counts = np.unique(train_y, return_counts=True)
    class_counts = [0] * n_classes
    for u, c in zip(unique, counts):
        class_counts[int(u)] = int(c)
    with open(os.path.join(OUT_DIR, 'train_class_counts.json'), 'w', encoding='utf-8') as f:
        json.dump(class_counts, f, indent=2)

    # Сохраняем маппинг классов (индекс → название планеты)
    idx_to_name = {str(v): k for k, v in label_mapping.items()}
    with open(os.path.join(OUT_DIR, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(idx_to_name, f, indent=2, ensure_ascii=False)

    # Топ-5 классов в валидации
    unique_v, counts_v = np.unique(valid_y, return_counts=True)
    top5 = sorted(zip(unique_v.tolist(), counts_v.tolist()), key=lambda x: -x[1])[:5]
    valid_top5 = {
        'labels': [idx_to_name.get(str(c), str(c)) for c, _ in top5],
        'values': [v for _, v in top5],
    }
    with open(os.path.join(OUT_DIR, 'valid_top5.json'), 'w', encoding='utf-8') as f:
        json.dump(valid_top5, f, indent=2, ensure_ascii=False)

    print('Готово. Модель и JSON сохранены в', OUT_DIR)


if __name__ == '__main__':
    main()
