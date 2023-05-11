

# Third-party imports
import numpy as np
import pandas as pd
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.cache import cache
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.views.decorators.cache import cache_page
from django.views.decorators.csrf import csrf_protect
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine, text
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, LSTM
from django.http import JsonResponse
from django.contrib.auth.models import Group
from django.core import serializers
from .models import CustomUser
from .models import CellData
# Local imports
from .forms import CustomUserCreationForm, LoginForm, CustomUserEditForm
from datetime import datetime, timedelta



###
import tensorflow as tf
import os
import json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau




def cell_data(request):
    # Get the date range from the request or default to the last 30 days
    end_date = request.GET.get('end_date', datetime.now().strftime("%Y-%m-%d"))
    start_date = request.GET.get('start_date', (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))

    data = CellData.objects.filter(Time__range=(start_date, end_date)).values('Time', 'FT_AVERAGE_NB_OF_USERS', 'FT_4G_LTE_DL_TRAFFIC_VOLUME')
    response_data = list(data)
    return JsonResponse(response_data, safe=False)








def add_user(request):
    if not request.user.is_admin:
        return JsonResponse({"error": "Permission denied"}, status=403)
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Add any necessary default user group here
            messages.success(request, f'User {user.username} added successfully!')
            return redirect('admin_dashboard')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'accounts/add_user.html', {'form': form})

def delete_user(request, user_id):
    if not request.user.is_admin:
        return JsonResponse({"error": "Permission denied"}, status=403)

    user = CustomUser.objects.get(id=user_id)
    if user:
        user.delete()
        messages.success(request, f'User {user.username} deleted successfully!')
        return JsonResponse({'status': 'success'})
    else:
        messages.error(request, f'User not found')
        return JsonResponse({'status': 'error'})





def edit_user(request, user_id):
    if not request.user.is_admin:
        return JsonResponse({"error": "Permission denied"}, status=403)

    user = CustomUser.objects.get(id=user_id)

    if request.method == 'POST':
        form = CustomUserEditForm(request.POST, instance=user)
        if form.is_valid():
            user = form.save()
            messages.success(request, f'User {user.username} updated successfully!')
            return redirect('admin_dashboard')
    else:
        form = CustomUserEditForm(instance=user)  # Populate the form with the user's data

    return render(request, 'accounts/edit_user.html', {'form': form})















def get_group_names(user):
    group_names = []
    if user.is_admin:
        group_names.append('Admin')
    if user.is_marketing:
        group_names.append('Marketing')
    if user.is_drs:
        group_names.append('DRS')
    if user.is_smc:
        group_names.append('SMC')
    if user.is_client_service:
        group_names.append('Client Service')
    return ', '.join(group_names)

@csrf_protect
@login_required
def get_users(request):
    if not request.user.is_admin:
        return JsonResponse({"error": "Permission denied"}, status=403)

    users = CustomUser.objects.all()
    user_list = []
    for user in users:
        user_list.append({
            'pk': user.pk,
            'email': user.email,
            'username': user.username,
            'group_names': get_group_names(user)
        })

    users_json = json.dumps(user_list)
    return JsonResponse(users_json, safe=False)


@csrf_protect
@login_required
def get_groups(request):
    if not request.user.is_admin:
        return JsonResponse({"error": "Permission denied"}, status=403)
    groups = Group.objects.all()
    groups_json = serializers.serialize('json', groups)
    return JsonResponse(groups_json, safe=False)



@csrf_protect
def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = True
            if form.cleaned_data['role'] == 'Admin':
                user.is_admin = True
                user.save()
            elif form.cleaned_data['role'] == 'Marketing':
                user.is_marketing = True
            elif form.cleaned_data['role'] == 'DRS':
                user.is_drs = True
            elif form.cleaned_data['role'] == 'SMC':
                user.is_smc = True
            elif form.cleaned_data['role'] == 'Client Service':
                user.is_client_service = True
            user.save()
            login(request, user)
            return redirect(reverse('dashboard'))
    else:
     form = CustomUserCreationForm()
    return render(request, 'accounts/register.html', {'form': form})







@csrf_protect
def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                if user.is_admin:
                    return redirect(reverse('admin_dashboard'))
                elif user.is_marketing:
                    return redirect(reverse('marketing_dashboard'))
                elif user.is_drs:
                    return redirect(reverse('drs_dashboard'))
                elif user.is_smc:
                    return redirect(reverse('smc_dashboard'))
                elif user.is_client_service:
                    return redirect(reverse('client_service_dashboard'))
            else:
                # Handle invalid login credentials
                error_message = 'Invalid email or password.'
                return render(request, 'accounts/login.html', {'form': form, 'error_message': error_message})
        else:
            # Handle invalid form input
            return render(request, 'accounts/login.html', {'form': form})
    else:
        form = LoginForm()
        return render(request, 'accounts/login.html', {'form': form})



@csrf_protect
@login_required
def dashboard(request):
    user = request.user
    if user.is_admin:
        return redirect(reverse('admin_dashboard'))
    elif user.is_marketing:
        return redirect(reverse('marketing_dashboard'))
    elif user.is_drs:
        return redirect(reverse('drs_dashboard'))
    elif user.is_smc:
        return redirect(reverse('smc_dashboard'))
    elif user.is_client_service:
        return redirect(reverse('client_service_dashboard'))
    else:
        return render('default_dashboard')


##################################
#class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1} completed, loss: {logs.get('loss')}")



#def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        X.append(dataset[i:(i+look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


#def generate_forecast():
    # Check if the model already exists and load it
    users_model_version = '1'
    users_model_path = f'./models/users/{users_model_version}'
    if os.path.exists(users_model_path):
        model = load_model(users_model_path)
    else:
        data = pd.read_csv('first-4G-cell_Query_Result_20230329130945660(Subreport 1).csv')
        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time', inplace=True)
    
    data = data[['FT_AVERAGE NB OF USERS (UEs RRC CONNECTED)']]

    # Scale the data using a MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create training and testing datasets
    look_back = 24
    train_X, train_Y = create_dataset(scaled_data[:int(len(scaled_data) * 0.8)], look_back)
    test_X, test_Y = create_dataset(scaled_data[int(len(scaled_data) * 0.8):], look_back)

    # Reshape the input data for the LSTM model
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    # Define the ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Train the model with the progress_callback and early_stopping
    progress_callback = TrainingProgressCallback()
    model.fit(train_X, train_Y, epochs=1, batch_size=1, verbose=0, callbacks=[progress_callback, early_stopping, reduce_lr], validation_data=(test_X, test_Y))

    # Make predictions using the trained model
    forecast = model.predict(test_X)

    # Invert scaling for predictions
    forecast_users = scaler.inverse_transform(forecast)
    actual_users = scaler.inverse_transform(data[int(len(scaled_data) * 0.8) + look_back:])


    # Save the trained model
    if not os.path.exists(users_model_path):
        os.makedirs(users_model_path)
    tf.saved_model.save(model, users_model_path)


    return {
        'actual_users': actual_users.flatten(),
        'forecast_users': forecast_users.flatten(),
    }
##################################






@csrf_protect
@login_required
def marketing_dashboard(request):
    if not request.user.is_drs:
        return redirect(reverse('dashboard'))
    
    return render(request, 'accounts/marketing_dashboard.html')

    




@csrf_protect
@login_required
def drs_dashboard(request):
    if not request.user.is_drs:
        return redirect(reverse('dashboard'))
    
    return render(request, 'accounts/drs_dashboard.html')


@csrf_protect
@login_required
def smc_dashboard(request):
    if not request.user.is_smc:
        return redirect(reverse('dashboard'))
    return render(request, 'accounts/smc_dashboard.html')


@csrf_protect
@login_required
def client_service_dashboard(request):
    if not request.user.is_client_service:
        return redirect(reverse('dashboard'))
    return render(request, 'accounts/client_service_dashboard.html')





@csrf_protect
@login_required
@cache_page(86400)  # Cache the view for 24 hours (86400 seconds)
def admin_dashboard(request):
    if not request.user.is_admin:
        return redirect(reverse('dashboard'))
    all_users = CustomUser.objects.all()
    all_groups = Group.objects.all()
    context = {'users': all_users, 'groups': all_groups}
    
    #forecast_results = generate_forecast()

    # Prepare the data for the template
    """user_data = [
        {'x': str(idx), 'y': value}
        for idx, value in forecast_results['actual_users'].iteritems()
    ]
    user_forecast = [
        {'x': str(idx), 'y': value[0]}
        for idx, value in enumerate(forecast_results['forecast_users'].tolist())
    ]
    traffic_data = [
        {'x': str(idx), 'y': value}
        for idx, value in forecast_results['actual_traffic'].iteritems()
    ]
    traffic_forecast = [
        {'x': str(idx), 'y': value[0]}
        for idx, value in enumerate(forecast_results['forecast_traffic'].tolist())
    ]

    context = {
        'user_data': user_data,
        'user_forecast': user_forecast,
        'traffic_data': traffic_data,
        'traffic_forecast': traffic_forecast,
    }
"""
    return render(request, 'accounts/admin_dashboard.html', context)

