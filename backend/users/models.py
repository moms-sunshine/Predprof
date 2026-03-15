# Модель пользователя с ролями: админ может только создавать пользователей,
# пользователь — загружать данные и смотреть аналитику.
from django.contrib.auth.models import AbstractUser
from django.db import models

ROLE_ADMIN = 'admin'
ROLE_USER = 'user'
ROLES = [(ROLE_ADMIN, 'Администратор'), (ROLE_USER, 'Пользователь')]


class User(AbstractUser):
    """Пользователь: логин, пароль, имя, фамилия и роль."""
    role = models.CharField(max_length=10, choices=ROLES, default=ROLE_USER)

    def is_admin_role(self):
        return self.role == ROLE_ADMIN
