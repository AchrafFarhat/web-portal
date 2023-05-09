
from django.contrib.auth.base_user import BaseUserManager
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.db import models
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX



class CustomUserQuerySet(models.QuerySet):
    # define your custom queryset methods here
    pass




class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, **extra_fields)
    
objects = CustomUserManager.from_queryset(CustomUserQuerySet)()

class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=30, unique=True)
    is_admin = models.BooleanField(default=False)
    is_marketing = models.BooleanField(default=False)
    is_drs = models.BooleanField(default=False)
    is_smc = models.BooleanField(default=False)
    is_client_service = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    objects = CustomUserManager()

    def __str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        return True

    def has_module_perms(self, app_label):
        return True
    


class CellData(models.Model):
    Time = models.DateTimeField()
    eNodeB_Name = models.CharField(max_length=100)
    Frequency_band = models.IntegerField()
    Cell_FDD_TDD_Indication = models.CharField(max_length=10)
    Cell_Name = models.CharField(max_length=100)
    Downlink_EARFCN = models.IntegerField()
    Downlink_bandwidth = models.CharField(max_length=50)
    LTECell_Tx_and_Rx_Mode = models.CharField(max_length=50)
    LocalCell_Id = models.IntegerField()
    eNodeB_Function_Name = models.CharField(max_length=100)
    Latitude = models.FloatField()
    Longitude = models.FloatField()
    Integrity = models.CharField(max_length=5)
    FT_AVE_4G_LTE_DL_USER_THRPUT_without_Last_TTI = models.FloatField()
    FT_AVERAGE_NB_OF_USERS = models.FloatField()
    FT_PHYSICAL_RESOURCE_BLOCKS_LOAD_DL = models.FloatField()
    FT_PHYSICAL_RESOURCE_BLOCKS_LOAD_UL = models.FloatField()
    FT_4G_LTE_DL_TRAFFIC_VOLUME = models.FloatField()
    FT_4G_LTE_DL_UL_TRAFFIC_VOLUME = models.FloatField()
    FT_4G_LTE_UL_TRAFFIC_VOLUME = models.FloatField()
    AVE_4G_LTE_DL_USER_THRPUT_without_Last_TTI = models.FloatField()
    Average_Nb_of_Used_PRB_for_SRB = models.FloatField()
    Average_Nb_of_PRB_used_per_Ue = models.FloatField()
    Average_Nb_of_e_RAB_per_UE = models.FloatField()
    FT_4G_LTE_CONGESTED_CELLS_RATE = models.CharField(max_length=10)
    FT_4G_LTE_CALL_SETUP_SUCCESS_RATE = models.FloatField()
    FT_4G_LTE_DROP_CALL_RATE_FOR_CARRIER_AGGREGGATION = models.FloatField()
    FT_4G_LTE_DROP_CALL_RATE = models.FloatField()
    FT_4G_LTE_VOLTE_TRAFFIC_VOLUME = models.FloatField()
    FT_CS_FALLBACK_SUCCESS_RATE = models.FloatField()
    FT_CS_FALLBACK_TO_WCDMA_RATIO = models.FloatField()
    FT_S1_SUCCESS_RATE = models.FloatField()
    Average_number_of_activated_UEs = models.FloatField()
    Average_number_of_users_with_data_in_the_buffer = models.FloatField()
    Connected_user_license = models.FloatField()
    Maximum_number_of_activated_UEs = models.IntegerField()
    Maximum_number_of_users_with_data_in_the_buffer = models.FloatField()
    PRBs = models.FloatField()
    PUCCH_and_SRS_resources = models.FloatField()
    RRC_SetupFail = models.IntegerField()
    Service_drop_rate = models.FloatField()
    Service_Integrity = models.FloatField()

    def __str__(self):
        return f"{self.Cell_Name} - {self.Time}"


    