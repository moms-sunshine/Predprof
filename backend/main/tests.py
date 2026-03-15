from django.test import TestCase, Client
from django.urls import reverse
from users.models import User


class ProfileTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='user1', password='pass', first_name='Алексей', last_name='Иванов', role='user')

    def test_user_sees_profile_after_login(self):
        self.client.login(username='user1', password='pass')
        resp = self.client.get(reverse('main:profile'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Алексей')
        self.assertContains(resp, 'Иванов')

    def test_anonymous_redirected_to_login(self):
        resp = self.client.get(reverse('main:profile'))
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.url.startswith('/') or resp.get('Location', '').startswith('/'))


class AnalyticsTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='user1', password='pass', role='user')

    def test_analytics_page_opens_for_user(self):
        self.client.login(username='user1', password='pass')
        resp = self.client.get(reverse('main:analytics'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Графики')
