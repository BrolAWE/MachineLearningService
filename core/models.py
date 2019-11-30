from django.db import models


# Create your models here.

class Train(models.Model):
    PassengerId = models.IntegerField(primary_key=True, null=True, blank=True)
    Survived = models.IntegerField(null=True, blank=True)
    Pclass = models.IntegerField(null=True, blank=True)
    Name = models.CharField(max_length=100, null=True, blank=True)
    Sex = models.CharField(max_length=100, null=True, blank=True)
    Age = models.FloatField(null=True, blank=True)
    SipSp = models.IntegerField(null=True, blank=True)
    Parch = models.IntegerField(null=True, blank=True)
    Ticket = models.CharField(max_length=100, null=True, blank=True)
    Fare = models.FloatField(null=True, blank=True)
    Cabin = models.CharField(max_length=100, null=True, blank=True)
    Embarked = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return "{0}".format(self.pk)

    class Meta:
        verbose_name = "Раздел"
        verbose_name_plural = "Разделы"
