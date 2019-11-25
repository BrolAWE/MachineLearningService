from django.http import HttpResponse
from django.shortcuts import render
import numpy as np

# Create your views here.
from sklearn.linear_model import LinearRegression

x_train = np.array([
    [1, 10, 20170530, 10],
    [2, 120, 20170530, 10],
    [3, 300, 20170530, 10],
    [1, 16, 20170630, 11],
    [2, 130, 20170630, 11],
    [3, 320, 20170630, 11],
    [1, 20, 20170730, 12],
    [2, 142, 20170730, 12],
    [3, 340, 20170730, 12],
    [1, 25, 20170830, 13],
    [2, 150, 20170830, 13],
    [3, 330, 20170830, 13],
    [1, 20, 20170930, 14],
    [2, 110, 20170930, 14],
    [3, 360, 20170930, 14],
])

y_train = np.array([30, 160, 600, 32, 172, 600, 34, 169, 660, 36, 180, 700, 38, 150, 700])


def index(request):
    return HttpResponse("Привет")


def regr(request):
    p1 = float(request.GET['p1'])
    p2 = float(request.GET['p2'])
    p3 = float(request.GET['p3'])
    p4 = float(request.GET['p4'])
    X_test = np.array([[p1, p2, p3, p4]])
    Y_test = np.array([40])

    clf = LinearRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(X_test)

    return HttpResponse(y_pred)
