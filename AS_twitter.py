# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 13:22:01 2023

@author: jjmm08
"""

#######################################################################################
# 1. Tratamiento de datos: se importa los archivos que nos servirán
#                            en la construcción del modelo. Para esto, se utiliza 
#                            el archivo limpieza_tweet.py el cual se encarga de 
#                            realizar el tratamiento de los datos, quitar
#                            caracteres especiales, convertir a minúsuculas, etc. 
#
# 2. Machine Learning Model: una vez realizado el tratamiento de los datos,
#                            se procede a entrenar los distintos modelos.
#
# 3. Resultados del testeo: muestran los resultados que se obtienen al evaluar la data de test
#
# 4. Exportar el modelo de análisis de sentimiento del cual se obtiene mejores resultados 
########################################################################################

############# 1. Tratamiento de datos

import pandas as pd
from limpieza_tweet import limpieza

with open('data_model\\tweets_pos.txt', 'r', encoding='utf-8') as archivo:
    contenido_pos = archivo.readlines()
    
with open('data_model\\tweets_neg.txt', 'r', encoding='utf-8') as archivo:
    contenido_neg = archivo.readlines()

with open('data_model\\tweets_neu.txt', 'r', encoding='utf-8') as archivo:
    contenido_neu = archivo.readlines()
    
df_pos = pd.DataFrame({'tweet': contenido_pos})

df_neg = pd.DataFrame({'tweet': contenido_neg})
df_neg = df_neg.sample(n=50000, random_state=42)  


df_neu = pd.DataFrame({'tweet': contenido_neu})
df_neu = df_neu.sample(n=50000, random_state=42)  

df_pos['etiqueta'] =  1 
df_neu['etiqueta'] =  0
df_neg['etiqueta'] = -1

dataframes = [df_pos, df_neu, df_neg]
df_tweet = pd.concat(dataframes, ignore_index=True)

df_tweet_clean = df_tweet.sample(n=100000, random_state=42)
df_tweet_clean = limpieza(df_tweet_clean)
df_tweet_clean = df_tweet_clean[df_tweet_clean["tweet"]!=""]

############# 2. Machine Learning Model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(df_tweet_clean['tweet'], df_tweet_clean['etiqueta'],
                                                    test_size=0.2, random_state=7, stratify=df_tweet_clean['etiqueta'])


v = TfidfVectorizer()

X_train_normalized = v.fit_transform(X_train)
X_test_normalized = v.transform(X_test)

############ Random Forest

clf = RandomForestClassifier(max_depth=50) #n_estimators=100 por defecto

clf.fit(X_train_normalized, y_train)
y_pred = clf.predict(X_test_normalized)
y_pred_train = clf.predict(X_train_normalized)


f1_macro_train = f1_score(y_train, y_pred_train, average='macro')
f1_macro_test = f1_score(y_test, y_pred, average='macro')

cf_matrix = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])



############ MultinomialNB


nb_clf = MultinomialNB()
nb_clf.fit(X_train_normalized, y_train)

y_pred = nb_clf.predict(X_test_normalized)
y_pred_train = nb_clf.predict(X_train_normalized)

f1_macro_test_mnb = f1_score(y_test, y_pred, average='macro')
f1_macro_train_mnb = f1_score(y_train, y_pred_train, average='macro')


############ Regresión lineal


logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logistic_model.fit(X_train_normalized, y_train)

y_pred = logistic_model.predict(X_test_normalized)
y_pred_train = logistic_model.predict(X_train_normalized)

accuracy = accuracy_score(y_test, y_pred)
f1_macro_test_rl = f1_score(y_test, y_pred, average='macro')
f1_macro_train_rl = f1_score(y_train, y_pred_train, average='macro')


############ LGBoost

y_train_cambio=y_train.replace(-1,2)
y_test_cambio=y_test.replace(-1,2)

train_data = lgb.Dataset(X_train_normalized, label=y_train_cambio,categorical_feature=[0,1,2])
params = {
    'objective': 'multiclass',  # Problema de clasificación multiclase
    'num_class': 3,  # Número de clases en tu problema
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',  # Métrica de evaluación para clasificación multiclase
    'num_leaves': 200,
    'learning_rate': 0.15,
    'feature_fraction': 0.2,
    'bagging_fraction': 0.9,
    'bagging_freq': 15,
    'verbose': 0
}

num_round = 200  # Número de iteraciones 
bst = lgb.train(params, train_data, num_round)

y_pred_prob_train = bst.predict(X_train_normalized)  # Probabilidades de pertenencia a cada clase
y_pred_class_train = np.argmax(y_pred_prob_train, axis=1)  # Clase predicha

y_pred_prob_test = bst.predict(X_test_normalized)  # Probabilidades de pertenencia a cada clase
y_pred_class_test = np.argmax(y_pred_prob_test, axis=1)  # Clase predicha

print("##############################")

############# 3. Resultados del testeo


print("\nRegresión lineal :",'\n')
y_pred = logistic_model.predict(X_test_normalized)
report = classification_report(y_test, y_pred)
print(report)

print("\nMultinomialNB :",'\n')
y_pred = nb_clf.predict(X_test_normalized)
report = classification_report(y_test, y_pred)
print(report)

print("\nRandom Forest :",'\n')
y_pred = clf.predict(X_test_normalized)
report = classification_report(y_test, y_pred)
print(report)

print("\nLGBoost test:",'\n')
y_pred = bst.predict(X_test_normalized)
y_pred_arg = np.argmax(y_pred, axis=1) 
y_test_cambio = np.where(y_test_cambio == 2, -1, y_test_cambio)
y_pred_arg = np.where(y_pred_arg == 2, -1, y_pred_arg)
report = classification_report(y_test_cambio, y_pred_arg)
print(report)


######## 4. Exportar el modelo de análisis de sentimiento


import joblib

saved_objects = {
    'tfidf_vectorizer': v,
    'logistic_model': logistic_model
}

joblib.dump(saved_objects, 'tfidf_logistic_model.pkl')