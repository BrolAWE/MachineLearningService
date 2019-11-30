from django.http import HttpResponse
from django.shortcuts import render
import numpy as np

from django.views.decorators.csrf import csrf_exempt
from neupy.algorithms import PNN
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from spyne import Float
from spyne.application import Application
from spyne.decorator import rpc
from spyne.model.primitive import Unicode, Integer
from spyne.protocol.soap import Soap11
from spyne.server.django import DjangoApplication
from spyne.service import ServiceBase

# Create your views here.
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

titanic_data = pd.read_csv('train.csv')
X_train = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
X_train = pd.get_dummies(X_train)
X_train = X_train.fillna({'Age': X_train.Age.median()})
y_train = titanic_data.Survived


def index(request):
    titanic_data = pd.read_csv('train.csv')
    X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    X = pd.get_dummies(X)
    X = X.fillna({'Age': X.Age.median()})
    y = titanic_data.Survived
    clf_rf = RandomForestClassifier()
    parametrs = {'n_estimators': [10, 20, 30], 'max_depth': [2, 5, 7, 10]}
    grid_search_cv_clf = GridSearchCV(clf_rf, parametrs, cv=5)
    grid_search_cv_clf.fit(X, y)
    X_predict = np.array([[3, 22, 1, 0, 7, 0, 1, 0, 0, 1]])
    y_predict = grid_search_cv_clf.predict(X_predict)
    print(X.iloc[0])
    return HttpResponse(y_predict)


def regr(request):
    p1 = float(request.GET['p1'])
    p2 = float(request.GET['p2'])
    p3 = float(request.GET['p3'])
    p4 = float(request.GET['p4'])
    X_test = np.array([[p1, p2, p3, p4]])

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return HttpResponse(y_pred)


class SoapService(ServiceBase):
    @rpc(Unicode(nillable=False), Unicode(nillable=False), _returns=Unicode)
    def hello(ctx, name1, name2):
        return name1 + "," + name2

    @rpc(Integer(nillable=False), Integer(nillable=False), _returns=Integer)
    def sum(ctx, a, b):
        return int(a + b)

    # Решающие деревья (Классификация)

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def tree(ctx, Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=100, min_samples_leaf=10)
        clf.fit(X_train, y_train)
        X_test = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S]])
        y_predict = clf.predict(X_test)

        return float(y_predict)

    # Ансамбль решающих деревьев

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def tree_ensemble(ctx, Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S):
        clf_rf = RandomForestClassifier()
        parametrs = {'n_estimators': [10, 20, 30], 'max_depth': [2, 5, 7, 10]}
        grid_search_cv_clf = GridSearchCV(clf_rf, parametrs, cv=5)
        grid_search_cv_clf.fit(X_train, y_train)
        X_test = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S]])
        y_predict = grid_search_cv_clf.predict(X_test)

        return float(y_predict)

    # Линейная регрессия

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def regr(ctx, Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S):
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        X_test = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S]])
        y_predict = clf.predict(X_test)

        return float(y_predict)

    # Ближайший сосед

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def neighbors(ctx, Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S):
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
        clf.fit(X_train, y_train)
        X_test = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S]])
        y_predict = clf.predict(X_test)

        return float(y_predict)

    # Кластеризация

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def neighbors(ctx, Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S):
        clf = KMeans(n_clusters=2)
        clf.fit(X_train, y_train)
        X_test = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S]])
        y_predict = clf.predict(X_test)

        return float(y_predict)

    # Вероятностная нейронная сеть

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def neuro(ctx, Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S):
        clf = PNN(verbose=False, std=10)
        clf.fit(X_train, y_train)
        X_test = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S]])
        y_predict = clf.predict(X_test)

        return float(y_predict)


soap_app = Application(
    [SoapService],
    tns='django.soap.machinelearningservice',
    in_protocol=Soap11(validator='lxml'),
    out_protocol=Soap11(),
)

django_soap_application = DjangoApplication(soap_app)
my_soap_application = csrf_exempt(django_soap_application)
