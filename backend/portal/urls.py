from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('', views.login_view, name='login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('admin1/', views.admin_dashboard, name='admin_dashboard'),
    path('marketing/', views.marketing_dashboard, name='marketing_dashboard'),
    path('drs/dashboard/', views.drs_dashboard, name='drs_dashboard'),
    path('smc/dashboard/', views.smc_dashboard, name='smc_dashboard'),
    path('client-service/dashboard/', views.client_service_dashboard, name='client_service_dashboard'),
    path('get_users/', views.get_users, name='get_users'),
    path('get_groups/', views.get_groups, name='get_groups'),
    path('add_user/', views.add_user, name='add_user'),
    path('delete_user/<int:user_id>/', views.delete_user, name='delete_user'),
    path('cell_data/', views.cell_data, name='cell_data'),
    #path('forecasting/dashboard/', views.forecasting_view, name='forecasting'),
]