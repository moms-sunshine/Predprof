# Хранение результатов проверки модели на загруженном тестовом наборе
from django.db import models
from django.conf import settings


class TestResult(models.Model):
    """Результат проверки модели на одном загруженном тестовом наборе."""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='test_results')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # Точность и потери на тестовом наборе
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    # Файл загружен пользователем (опционально храним путь)
    file_name = models.CharField(max_length=255, default='')
    # JSON с данными для графиков: точность по записям, предсказания по классам и т.д.
    chart_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-uploaded_at']
