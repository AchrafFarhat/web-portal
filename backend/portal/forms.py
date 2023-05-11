from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser, CellData

class CustomUserCreationForm(UserCreationForm):
    role_choices = [
        ('Admin', 'Admin'),
        ('Marketing', 'Marketing'),
        ('DRS', 'DRS'),
        ('SMC', 'SMC'),
        ('Client Service', 'Client Service'),
    ]
    role = forms.ChoiceField(choices=role_choices)

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2', 'role']


     
class LoginForm(forms.Form):
    email = forms.EmailField(label='Email address', required=True)
    password = forms.CharField(label='Password', widget=forms.PasswordInput(), required=True)


class CellDataForm(forms.ModelForm):
    Time = forms.DateTimeField(widget=forms.DateTimeInput(attrs={'type': 'datetime-local'}))
    
    class Meta:
        model = CellData
        fields = ['Time']


class CustomUserEditForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ['email', 'username','groups']