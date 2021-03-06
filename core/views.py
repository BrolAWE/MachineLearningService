from django.http import HttpResponse
import numpy as np

from django.views.decorators.csrf import csrf_exempt
from neupy.algorithms import PNN
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from spyne import Float
from spyne.application import Application
from spyne.decorator import rpc
from spyne.protocol.soap import Soap11
from spyne.server.django import DjangoApplication
from spyne.service import ServiceBase

# Create your views here.
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from core.models import Train

titanic_data = pd.DataFrame(list(Train.objects.all().values()))
X_train = titanic_data.drop(['passenger_id', 'survived', 'name', 'ticket', 'cabin'], axis=1)
X_train = pd.get_dummies(X_train)
X_train = X_train.fillna({'age': X_train.age.median()})
X_train = X_train.drop(['embarked_'], axis=1)
y_train = titanic_data.survived


def index(request):
    return HttpResponse("Привет")


class SoapService(ServiceBase):
    # Решающие деревья (Классификация)

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def tree(self, p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=100, min_samples_leaf=10)
        clf.fit(X_train, y_train)
        x_test = np.array(
            [[p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s]])
        y_predict = clf.predict(x_test)

        return float(y_predict)

    # Ансамбль решающих деревьев

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def tree_ensemble(self, p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q,
                      embarked_s):
        clf_rf = RandomForestClassifier()
        parameters = {'n_estimators': [10, 20, 30], 'max_depth': [2, 5, 7, 10]}
        grid_search_cv_clf = GridSearchCV(clf_rf, parameters, cv=5)
        grid_search_cv_clf.fit(X_train, y_train)
        x_test = np.array(
            [[p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s]])
        y_predict = grid_search_cv_clf.predict(x_test)

        return float(y_predict)

    # Логистическая регрессия

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def regression(self, p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s):
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        x_test = np.array(
            [[p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s]])
        y_predict = clf.predict(x_test)

        return float(y_predict)

    # Ближайший сосед

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def neighbors(self, p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s):
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
        clf.fit(X_train, y_train)
        x_test = np.array(
            [[p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s]])
        y_predict = clf.predict(x_test)

        return float(y_predict)

    # Кластеризация

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def ml(self, p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s):
        clf = KMeans(n_clusters=2)
        clf.fit(X_train, y_train)
        x_test = np.array(
            [[p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s]])
        y_predict = clf.predict(x_test)

        return float(y_predict)

    # Вероятностная нейронная сеть

    @rpc(Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), Float(nillable=False), Float(nillable=False),
         Float(nillable=False), Float(nillable=False), _returns=Float)
    def neuron(self, p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s):
        clf = PNN(verbose=False, std=10)
        clf.fit(X_train, y_train)
        x_test = np.array(
            [[p_class, age, sib_sp, par_ch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s]])
        y_predict = clf.predict(x_test)

        return float(y_predict)


soap_app = Application(
    [SoapService],
    tns='django.soap.machine_learning_service',
    in_protocol=Soap11(validator='lxml'),
    out_protocol=Soap11(),
)

django_soap_application = DjangoApplication(soap_app)
my_soap_application = csrf_exempt(django_soap_application)
