# Generated by Django 2.2.7 on 2019-12-02 22:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_auto_20191203_0150'),
    ]

    operations = [
        migrations.AlterField(
            model_name='train',
            name='age',
            field=models.FloatField(blank=True, null=True),
        ),
    ]