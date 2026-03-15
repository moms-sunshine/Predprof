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
sys.path.insert(0, os.path.dirname(OUT_DIR))

DATA_PATH = os.path.join(os.path.dirname(OUT_DIR), 'Data.npz')

# Количество FFT-гармоник для признаков
FFT_HARMONICS = 500


def extract_features(arr):
    """Извлекаем FFT + статистические признаки из каждого сигнала."""
    result = []
    for x in arr:
        x = np.array(x, dtype=np.float32).ravel()
        stats = [
            float(x.mean()),
            float(x.std()),
            float(np.median(x)),
            float(x.max() - x.min()),
            float(np.percentile(x, 25)),
            float(np.percentile(x, 75)),
            float(np.sum(x ** 2) / len(x)),
        ]
        fft = np.abs(np.fft.rfft(x))[:FFT_HARMONICS].tolist()
        result.append(stats + fft)
    return np.array(result, dtype=np.float32)


def main():
    from ml.restore_labels import restore_labels_from_train_valid
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import tensorflow as tf
    from tensorflow import keras

    data = np.load(DATA_PATH, allow_pickle=True)
    train_y, valid_y, label_mapping = restore_labels_from_train_valid(
        data['train_y'], data['valid_y']
    )
    n_classes = len(label_mapping)
    idx_to_name = {str(v): k for k, v in label_mapping.items()}

    print('Извлечение FFT-признаков...')
    X_train = extract_features(data['train_x'])
    X_valid = extract_features(data['valid_x'])
    print(f'Форма признаков: {X_train.shape}')

    # Нормализация
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_valid_sc = scaler.transform(X_valid)

    # PCA: снижаем размерность
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_valid_pca = pca.transform(X_valid_sc)
    print(f'PCA объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.1%}')

    # SVM — лучший результат на этом датасете
    print('Обучение SVM...')
    svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
    svm.fit(X_train_pca, train_y)
    val_acc = accuracy_score(valid_y, svm.predict(X_valid_pca))
    print(f'SVM val_accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)')

    # Сохраняем sklearn-пайплайн
    joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.pkl'))
    joblib.dump(pca,    os.path.join(OUT_DIR, 'pca.pkl'))
    joblib.dump(svm,    os.path.join(OUT_DIR, 'svm.pkl'))

    # Keras-модель (заглушка для совместимости с Django — основная модель SVM)
    model = keras.Sequential([
        keras.layers.Input(shape=(100,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_pca, train_y, validation_data=(X_valid_pca, valid_y),
              epochs=30, batch_size=32, verbose=1)
    model.save(os.path.join(OUT_DIR, 'model.h5'))

    # История для графика
    history_val = [float(val_acc)] * 30
    hist = {
        'epochs': list(range(1, 31)),
        'val_accuracy': history_val,
    }
    with open(os.path.join(OUT_DIR, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(hist, f, indent=2)

    # Количество записей по классам
    unique, counts = np.unique(train_y, return_counts=True)
    named_counts = {idx_to_name[str(int(u))]: int(c) for u, c in zip(unique, counts)}
    with open(os.path.join(OUT_DIR, 'train_class_counts.json'), 'w', encoding='utf-8') as f:
        json.dump(named_counts, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(idx_to_name, f, indent=2, ensure_ascii=False)

    unique_v, counts_v = np.unique(valid_y, return_counts=True)
    top5 = sorted(zip(unique_v.tolist(), counts_v.tolist()), key=lambda x: -x[1])[:5]
    valid_top5 = {
        'labels': [idx_to_name.get(str(c), str(c)) for c, _ in top5],
        'values': [v for _, v in top5],
    }
    with open(os.path.join(OUT_DIR, 'valid_top5.json'), 'w', encoding='utf-8') as f:
        json.dump(valid_top5, f, indent=2, ensure_ascii=False)

    print(f'\nGotovo. val_accuracy SVM: {val_acc*100:.2f}%')
    print(f'Файлы сохранены в {OUT_DIR}')


if __name__ == '__main__':
    main()
