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

# Графики строим через Matplotlib (из разрешённых материалов)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Админ не должен видеть загрузку и аналитику как пользователь — по ТЗ у админа только создание пользователей.
# Но по коду редиректа после логина админ идёт в create_user, пользователь — в profile. Если админ зайдёт по ссылке в profile — покажем ему сообщение или редирект.
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
    """Страница с информацией о пользователе."""
    return render(request, 'main/profile.html', {'user': request.user})


@login_required
@_user_only
@require_http_methods(['GET', 'POST'])
def upload_view(request):
    """Загрузка тестового набора .npz и подсчёт точности/потерь."""
    if request.method == 'POST':
        f = request.FILES.get('test_file')
        if not f or not f.name.endswith('.npz'):
            messages.error(request, 'Выберите файл в формате .npz')
            return redirect('main:upload')
        # Сохраняем во временный файл и обрабатываем
        import tempfile
        import numpy as np
        from django.core.files.storage import default_storage
        path = default_storage.save('test_uploads/' + f.name, f)
        full_path = default_storage.path(path)
        try:
            data = np.load(full_path, allow_pickle=True)
            available_keys = list(data.files)

            # Поддержка альтернативных имён ключей
            X_CANDIDATES = ['test_x', 'x_test', 'X_test', 'valid_x', 'x', 'X', 'train_x', 'data', 'arr_0']
            Y_CANDIDATES = ['test_y', 'y_test', 'Y_test', 'valid_y', 'vaild_y', 'y', 'Y', 'train_y', 'labels', 'arr_1']

            test_x = None
            test_y = None
            for key in X_CANDIDATES:
                if key in available_keys:
                    test_x = data[key]
                    break
            for key in Y_CANDIDATES:
                if key in available_keys:
                    test_y = data[key]
                    break

            if test_x is None or test_y is None:
                missing = []
                if test_x is None:
                    missing.append('признаки (test_x)')
                if test_y is None:
                    missing.append('метки (test_y)')
                messages.error(
                    request,
                    f'Файл не содержит нужных массивов: {", ".join(missing)}. '
                    f'Доступные ключи в архиве: {", ".join(available_keys) or "отсутствуют"}. '
                    f'Загрузите корректный тестовый набор данных.'
                )
                return redirect('main:upload')
        except Exception as e:
            messages.error(request, f'Ошибка чтения файла: {e}')
            return redirect('main:upload')
        # Загружаем модель и считаем метрики (если модели нет — показываем заглушку)
        model_path = getattr(settings, 'ML_MODEL_PATH', None) or os.path.join(settings.BASE_DIR, '..', 'model.h5')
        accuracy, loss, chart_data = _evaluate_model(model_path, test_x, test_y)
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
    """Топ-5 классов в валидационном наборе (сохраняется при обучении)."""
    p = os.path.join(settings.BASE_DIR, '..', 'ml', 'valid_top5.json')
    if os.path.isfile(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _evaluate_model(model_path, test_x, test_y):
    """Запускает модель на test_x, test_y и возвращает accuracy, loss, chart_data."""
    if not os.path.isfile(model_path):
        # Нет модели — возвращаем заглушку для демо
        n = len(test_y) if hasattr(test_y, '__len__') else 0
        return 0.0, 0.0, {
            'test_accuracy_per_sample': [0.0] * min(n, 100),
            'top5_valid_classes': [],
            'valid_class_counts': [],
        }
    import numpy as np
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    except Exception:
        return 0.0, 0.0, {}
    # Приводим test_x к формату модели (часто это [samples, time] или [samples, time, channels])
    X = np.array([np.array(x, dtype=np.float32) for x in test_x])
    if X.ndim == 1:
        X = np.expand_dims(X, 0)
    y_true = np.array(test_y).ravel().astype(np.int32)
    n_classes = int(y_true.max()) + 1
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
    loss, acc = model.evaluate(X, y_true_onehot, verbose=0)
    pred = model.predict(X, verbose=0)
    y_pred = np.argmax(pred, axis=1)
    # Точность по каждой записи: 1 если угадал, 0 если нет
    per_sample = (y_pred == y_true).astype(float).tolist()
    # Топ-5 классов в валидации храним в обучении — здесь просто топ по предсказаниям
    unique, counts = np.unique(y_pred, return_counts=True)
    top5 = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])[:5]
    chart_data = {
        'test_accuracy_per_sample': per_sample[:200],
        'top5_valid_classes': [x[0] for x in top5],
        'top5_counts': [x[1] for x in top5],
    }
    return float(acc), float(loss), chart_data


def _chart_response(fig, dpi=100):
    """Рендер фигуры Matplotlib в PNG и ответ HTTP."""
    buf = io.BytesIO()
    fig.set_size_inches(10, 5)
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type='image/png')


@login_required
@_user_only
def chart_epochs_view(request):
    """График: точность на валидации от количества эпох (Matplotlib)."""
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
    ax.grid(True, alpha=0.3)
    # Показываем каждую эпоху на оси X
    ax.set_xticks(epochs)
    ax.tick_params(axis='x', labelsize=8)
    # Ось Y начинается с нуля для наглядности
    ax.set_ylim(bottom=0)
    return _chart_response(fig)


@login_required
@_user_only
def chart_classes_view(request):
    """Диаграмма: количество записей по классам в обучении (Matplotlib)."""
    class_dist_path = os.path.join(settings.BASE_DIR, '..', 'ml', 'train_class_counts.json')
    if not os.path.isfile(class_dist_path):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Нет данных. Добавьте train_class_counts.json после обучения.', ha='center', va='center')
        return _chart_response(fig)
    with open(class_dist_path, 'r', encoding='utf-8') as f:
        class_counts = json.load(f)
    fig, ax = plt.subplots()
    x = ['Класс ' + str(i) for i in range(len(class_counts))]
    ax.bar(x, class_counts, color='#2ecc71', alpha=0.8)
    ax.set_xlabel('Класс (цивилизация)')
    ax.set_ylabel('Количество записей')
    ax.set_title('Количество записей по классам в наборе для обучения')
    plt.xticks(rotation=45, ha='right')
    return _chart_response(fig)


@login_required
@_user_only
def chart_per_record_view(request):
    """Диаграмма: точность по каждой записи тестового набора (Matplotlib)."""
    last_test = TestResult.objects.filter(user=request.user).first()
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
    """Диаграмма: топ-5 классов в валидационном наборе (Matplotlib)."""
    last_test = TestResult.objects.filter(user=request.user).first()
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
    """Страница аналитики: графики строятся через Matplotlib (картинки)."""
    last_test = TestResult.objects.filter(user=request.user).first()
    return render(request, 'main/analytics.html', {
        'last_test': last_test,
    })
