from django.db import models


# Create your models here.

class Train(models.Model):
    passenger_id = models.IntegerField(primary_key=True)
    survived = models.IntegerField(null=True, blank=True)
    p_class = models.IntegerField(null=True, blank=True)
    name = models.CharField(max_length=100, null=True, blank=True)
    sex = models.CharField(max_length=100, null=True, blank=True)
    age = models.FloatField(null=True, blank=True)
    sip_sp = models.IntegerField(null=True, blank=True)
    parch = models.IntegerField(null=True, blank=True)
    ticket = models.CharField(max_length=100, null=True, blank=True)
    fare = models.FloatField(null=True, blank=True)
    cabin = models.CharField(max_length=100, null=True, blank=True)
    embarked = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return "{0}".format(self.pk)
