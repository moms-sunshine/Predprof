from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

User = get_user_model()


class Command(BaseCommand):
    help = 'Создаёт администратора (логин admin, пароль admin) для первого входа.'

    def handle(self, *args, **options):
        if User.objects.filter(username='admin').exists():
            self.stdout.write('Пользователь admin уже есть.')
            return
        user = User.objects.create_user(username='admin', password='admin', role='admin', first_name='Админ', last_name='Системы')
        self.stdout.write(self.style.SUCCESS('Администратор создан: логин admin, пароль admin'))
