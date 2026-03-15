from django.urls import path
from . import views

app_name = 'main'

urlpatterns = [
    path('', views.profile_view, name='profile'),
    path('upload/', views.upload_view, name='upload'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('analytics/chart-epochs/', views.chart_epochs_view, name='chart_epochs'),
    path('analytics/chart-classes/', views.chart_classes_view, name='chart_classes'),
    path('analytics/chart-per-record/', views.chart_per_record_view, name='chart_per_record'),
    path('analytics/chart-top5/', views.chart_top5_view, name='chart_top5'),
]
