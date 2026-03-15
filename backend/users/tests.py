from django.test import TestCase, Client
from django.urls import reverse
from users.models import User


class LoginTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='test123', role='user')

    def test_login_page_opens(self):
        resp = self.client.get(reverse('users:login'))
        self.assertEqual(resp.status_code, 200)

    def test_login_redirects_user_to_profile(self):
        resp = self.client.post(reverse('users:login'), {'username': 'testuser', 'password': 'test123'})
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.url.endswith('/profile/'))

    def test_wrong_password_fails(self):
        resp = self.client.post(reverse('users:login'), {'username': 'testuser', 'password': 'wrong'})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Неверный')


class CreateUserTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create_user(username='admin', password='admin', role='admin')
        self.client.login(username='admin', password='admin')

    def test_admin_sees_create_form(self):
        resp = self.client.get(reverse('users:create_user'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Имя')
        self.assertContains(resp, 'Фамилия')

    def test_admin_creates_user(self):
        resp = self.client.post(reverse('users:create_user'), {
            'first_name': 'Иван',
            'last_name': 'Петров',
            'username': 'ivan',
            'password': 'ivan123',
        })
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(User.objects.filter(username='ivan').exists())
        new_user = User.objects.get(username='ivan')
        self.assertEqual(new_user.first_name, 'Иван')
        self.assertEqual(new_user.last_name, 'Петров')
        self.assertEqual(new_user.role, 'user')
