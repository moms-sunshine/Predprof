from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from .models import User, ROLE_ADMIN


def login_view(request):
    """Страница входа."""
    if request.user.is_authenticated:
        if request.user.is_admin_role():
            return redirect('users:create_user')
        return redirect('main:profile')
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        if not username or not password:
            messages.error(request, 'Введите логин и пароль.')
            return render(request, 'users/login.html')
        user = authenticate(request, username=username, password=password)
        if user is None:
            messages.error(request, 'Неверный логин или пароль.')
            return render(request, 'users/login.html')
        login(request, user)
        if user.is_admin_role():
            return redirect('users:create_user')
        return redirect('main:profile')
    return render(request, 'users/login.html')


@login_required
@require_http_methods(['GET', 'POST'])
def logout_view(request):
    logout(request)
    return redirect('users:login')


@login_required
@require_http_methods(['GET', 'POST'])
def create_user_view(request):
    """Только для админа: создание нового пользователя (имя, фамилия, логин, пароль)."""
    if not request.user.is_admin_role():
        messages.error(request, 'Доступ только для администратора.')
        return redirect('main:profile')
    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        if not first_name or not last_name:
            messages.error(request, 'Заполните имя и фамилию.')
            return render(request, 'users/create_user.html')
        if not username:
            messages.error(request, 'Введите логин.')
            return render(request, 'users/create_user.html')
        if not password or len(password) < 6:
            messages.error(request, 'Пароль не менее 6 символов.')
            return render(request, 'users/create_user.html')
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Такой логин уже есть.')
            return render(request, 'users/create_user.html')
        user = User.objects.create_user(
            username=username,
            password=password,
            first_name=first_name,
            last_name=last_name,
            role='user'
        )
        messages.success(request, f'Пользователь {username} создан.')
        return redirect('users:create_user')
    return render(request, 'users/create_user.html')
