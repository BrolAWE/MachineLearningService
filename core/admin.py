from django.contrib import admin

# Register your models here.
from core.models import Train


@admin.register(Train)
class SubsectionAdmin(admin.ModelAdmin):
    pass
