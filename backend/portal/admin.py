from django.contrib import admin
from .models import CustomUser, CellData

# Register your models here.
admin.site.register(CustomUser)
admin.site.register(CellData)
