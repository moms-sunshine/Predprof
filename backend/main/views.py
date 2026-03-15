import json
import os
import io
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.views.decorators.http import require_http_methods
from django.http import HttpResponse
from .models import TestResult

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _user_only(view_func):
    def wrap(request, *args, **kwargs):
        if request.user.is_authenticated and request.user.is_admin_role():
            messages.info(request, 'У администратора нет доступа к загрузке данных и аналитике.')
            return redirect('users:create_user')
        return view_func(request, *args, **kwargs)
    return wrap


@login_required
@_user_only
def profile_view(request):
    return render(request, 'main/profile.html', {'user': request.user})


@login_required
@_user_only
@require_http_methods(['GET', 'POST'])
def upload_view(request):
    if request.method == 'POST':
        f = request.FILES.get('test_file')
        if not f or not f.name.endswith('.npz'):
            messages.error(request, 'Выберите файл в формате .npz')
            return redirect('main:upload')

        import numpy as np
        from django.core.files.storage import default_storage

        path = default_storage.save('test_uploads/' + f.name, f)
        full_path = default_storage.path(path)

        try:
            data = np.load(full_path, allow_pickle=True)
            available_keys = list(data.files)
        except Exception as e:
            messages.error(request, f'Не удалось открыть файл: {e}')
            return redirect('main:upload')

        # Ищем массив признаков
        X_CANDIDATES = ['test_x', 'x_test', 'X_test', 'valid_x', 'train_x', 'x', 'X', 'data', 'arr_0']
        Y_CANDIDATES = ['test_y', 'y_test', 'Y_test', 'valid_y', 'vaild_y', 'train_y', 'y', 'Y', 'labels', 'arr_1']

        x_key = next((k for k in X_CANDIDATES if k in available_keys), None)
        y_key = next((k for k in Y_CANDIDATES if k in available_keys), None)

        if x_key is None or y_key is None:
            missing = []
            if x_key is None:
                missing.append('признаки (test_x/valid_x/train_x)')
            if y_key is None:
                missing.append('метки (test_y/valid_y/train_y)')
            messages.error(
                request,
                f'Файл не содержит нужных массивов: {", ".join(missing)}. '
                f'Доступные ключи: {", ".join(available_keys) or "отсутствуют"}.'
            )
            return redirect('main:upload')

        try:
            test_x = data[x_key]
            test_y = data[y_key]
        except Exception as e:
            messages.error(request, f'Ошибка чтения массивов из файла: {e}')
            return redirect('main:upload')

        # Если в файле есть train_y — строим маппинг как при обучении (train+valid вместе)
        # Это гарантирует совпадение нумерации классов с тем, что видела модель
        train_y_for_mapping = None
        if 'train_y' in available_keys and y_key != 'train_y':
            try:
                train_y_for_mapping = data['train_y']
            except Exception:
                pass

        model_path = getattr(settings, 'ML_MODEL_PATH', None) or os.path.join(settings.BASE_DIR, '..', 'ml', 'model.h5')
        accuracy, loss, chart_data = _evaluate_model(model_path, test_x, test_y, train_y_for_mapping)

        TestResult.objects.create(
            user=request.user,
            accuracy=accuracy,
            loss=loss,
            file_name=f.name,
            chart_data=chart_data or {}
        )
        messages.success(request, f'Файл обработан. Точность: {accuracy:.4f}, потери: {loss:.4f}')
        return redirect('main:analytics')

    return render(request, 'main/upload.html')


def _load_valid_top5():
    p = os.path.join(settings.BASE_DIR, '..', 'ml', 'valid_top5.json')
    if os.path.isfile(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _evaluate_model(model_path, test_x, test_y, train_y=None):
    import numpy as np
    if not os.path.isfile(model_path):
        n = len(test_y) if hasattr(test_y, '__len__') else 0
        return 0.0, 0.0, {
            'test_accuracy_per_sample': [0.0] * min(n, 100),
            'top5_valid_classes': [],
            'valid_class_counts': [],
        }
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    except Exception:
        return 0.0, 0.0, {}

    import joblib

    def _extract_fft_features(arr):
        result = []
        for x in arr:
            x = np.array(x, dtype=np.float32).ravel()
            stats = [x.mean(), x.std(), float(np.median(x)), float(x.max()-x.min()),
                     float(np.percentile(x, 25)), float(np.percentile(x, 75)),
                     float(np.sum(x**2)/len(x))]
            fft = np.abs(np.fft.rfft(x))[:500].tolist()
            result.append(stats + fft)
        return np.array(result, dtype=np.float32)

    ml_dir    = os.path.join(os.path.dirname(__file__), '..', '..', 'ml')
    scaler_path = os.path.join(ml_dir, 'scaler.pkl')
    pca_path    = os.path.join(ml_dir, 'pca.pkl')
    svm_path    = os.path.join(ml_dir, 'svm.pkl')
    has_pipeline = os.path.isfile(scaler_path) and os.path.isfile(pca_path)

    # Метки могут быть строками — конвертируем в целые числа
    # Маппинг строим по train+valid вместе, как при обучении
    y_raw = np.array(test_y).ravel()
    if y_raw.dtype.kind in ('U', 'S', 'O'):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ml'))
        try:
            from restore_labels import restore_labels, get_label_mapping
            if train_y is not None:
                # Единый маппинг по train+valid — точно как при обучении
                all_labels = np.concatenate([np.array(train_y).ravel(), y_raw])
                mapping = get_label_mapping(all_labels)
            else:
                mapping = get_label_mapping(y_raw)
            y_true, _ = restore_labels(y_raw, mapping)
        except Exception:
            unique = np.unique(y_raw)
            mapping = {str(u): i for i, u in enumerate(unique)}
            y_true = np.array([mapping[str(v)] for v in y_raw], dtype=np.int32)
    else:
        y_true = y_raw.astype(np.int32)

    # SVM-пайплайн (лучший результат на датасете)
    if has_pipeline and os.path.isfile(svm_path):
        scaler = joblib.load(scaler_path)
        pca    = joblib.load(pca_path)
        svm    = joblib.load(svm_path)
        X = _extract_fft_features(test_x)
        X = pca.transform(scaler.transform(X))
        y_pred = svm.predict(X)
        acc = float(np.mean(y_pred == y_true))
        per_sample = (y_pred == y_true).astype(float).tolist()
        unique, counts = np.unique(y_pred, return_counts=True)
        top5 = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])[:5]
        return acc, 0.0, {
            'test_accuracy_per_sample': per_sample[:200],
            'top5_valid_classes': [x[0] for x in top5],
            'top5_counts': [x[1] for x in top5],
        }

    # Fallback: Keras MLP
    X = np.array([np.array(x, dtype=np.float32).ravel() for x in test_x])
    if X.ndim == 1:
        X = np.expand_dims(X, 0)
    if has_pipeline:
        scaler = joblib.load(scaler_path)
        pca    = joblib.load(pca_path)
        X = pca.transform(scaler.transform(X))
    n_classes = model.output_shape[-1]
    y_true = np.clip(y_true, 0, n_classes - 1)
    loss, acc = model.evaluate(X, y_true, verbose=0)
    pred = model.predict(X, verbose=0)
    y_pred = np.argmax(pred, axis=1)
    per_sample = (y_pred == y_true).astype(float).tolist()
    unique, counts = np.unique(y_pred, return_counts=True)
    top5 = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])[:5]
    return float(acc), float(loss), {
        'test_accuracy_per_sample': per_sample[:200],
        'top5_valid_classes': [x[0] for x in top5],
        'top5_counts': [x[1] for x in top5],
    }


def _chart_response(fig, dpi=100):
    buf = io.BytesIO()
    fig.set_size_inches(10, 5)
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type='image/png')


@login_required
@_user_only
def chart_epochs_view(request):
    history_path = os.path.join(settings.BASE_DIR, '..', 'ml', 'training_history.json')
    if not os.path.isfile(history_path):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Нет данных. Запустите обучение в DataSphere.', ha='center', va='center')
        return _chart_response(fig)
    with open(history_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    epochs = data.get('epochs', list(range(1, len(data.get('val_accuracy', [])) + 1)))
    val_acc = data.get('val_accuracy', [])
    fig, ax = plt.subplots()
    ax.plot(epochs, val_acc, 'b-o', markersize=4)
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Точность на валидации')
    ax.set_title('Точность на валидации от количества эпох')
    ax.set_xticks(epochs)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    return _chart_response(fig)


@login_required
@_user_only
def chart_classes_view(request):
    class_dist_path = os.path.join(settings.BASE_DIR, '..', 'ml', 'train_class_counts.json')
    if not os.path.isfile(class_dist_path):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Нет данных. Добавьте train_class_counts.json после обучения.', ha='center', va='center')
        return _chart_response(fig)
    with open(class_dist_path, 'r', encoding='utf-8') as f:
        class_counts = json.load(f)
    fig, ax = plt.subplots()
    if isinstance(class_counts, dict):
        x = list(class_counts.keys())
        y = list(class_counts.values())
    else:
        x = [str(i) for i in range(len(class_counts))]
        y = class_counts
    ax.bar(x, y, color='#2ecc71', alpha=0.8)
    ax.set_xlabel('Класс (цивилизация)')
    ax.set_ylabel('Количество записей')
    ax.set_title('Количество записей по классам в наборе для обучения')
    plt.xticks(rotation=45, ha='right')
    return _chart_response(fig)


@login_required
@_user_only
def chart_per_record_view(request):
    last_test = TestResult.objects.filter(user=request.user).order_by('-id').first()
    per_sample = (last_test.chart_data or {}).get('test_accuracy_per_sample', []) if last_test else []
    fig, ax = plt.subplots()
    if not per_sample:
        ax.text(0.5, 0.5, 'Загрузите тестовый набор для отображения.', ha='center', va='center')
        return _chart_response(fig)
    ax.plot(range(1, len(per_sample) + 1), per_sample, 'purple', alpha=0.7)
    ax.set_xlabel('Номер записи')
    ax.set_ylabel('Верно (1) / Неверно (0)')
    ax.set_title('Точность по записям тестового набора')
    ax.grid(True, alpha=0.3)
    return _chart_response(fig)


@login_required
@_user_only
def chart_top5_view(request):
    last_test = TestResult.objects.filter(user=request.user).order_by('-id').first()
    chart = (last_test.chart_data or {}) if last_test else {}
    top5_labels = chart.get('top5_valid_classes', [])
    top5_counts = chart.get('top5_counts', [])
    if not top5_labels or not top5_counts:
        valid = _load_valid_top5()
        top5_labels = valid.get('labels', [])
        top5_values = valid.get('values', [])
        if not top5_labels:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Нет данных. Добавьте valid_top5.json или загрузите тест.', ha='center', va='center')
            return _chart_response(fig)
        top5_counts = top5_values
    else:
        top5_labels = ['Класс ' + str(c) for c in top5_labels]
    fig, ax = plt.subplots()
    ax.bar(top5_labels, top5_counts, color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Класс')
    ax.set_ylabel('Количество')
    ax.set_title('Топ-5 классов в валидационном наборе')
    plt.xticks(rotation=45, ha='right')
    return _chart_response(fig)


@login_required
@_user_only
def analytics_view(request):
    last_test = TestResult.objects.filter(user=request.user).order_by('-id').first()
    return render(request, 'main/analytics.html', {
        'last_test': last_test,
    })
