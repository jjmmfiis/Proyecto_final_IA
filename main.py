# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:30:24 2023

@author: jjmm08
"""


import tweepy
from password import consumer_key, consumer_secret, access_token_secret, access_token
import pandas as pd
from limpieza_tweet import limpieza
import pickle
import numpy as np
import random

def obtener_bdmusica():
    base_musica = pd.read_csv("tracklist2.csv",sep=";")
    base_musica["mapeo"] = base_musica["mapeo"].str.lower()
    return(base_musica)

def listar_canciones(df_musica,estado):
    df_musica_estado = df_musica[df_musica["mapeo"] == estado]
    df_musica_estado = df_musica_estado.loc[:,["Album name","Track"]]
    tamano_muestra = 20
    return(df_musica_estado.sample(n=tamano_muestra, random_state=42))


if __name__ == '__main__':

    # Utilizar claves de la API developer de twitter
    auth = tweepy.OAuth1UserHandler(
        consumer_key, consumer_secret, access_token, access_token_secret
    )
    
    api = tweepy.API(auth)
    
    # Obtener desde un txt el id de twitter del usuario
    with open('usuario.txt', 'r', encoding='utf-8') as archivo:
        id_user = archivo.read()
        
    user = api.get_user(screen_name=id_user)
    
    tweet,fecha = user.status._json['text'],user.status._json['created_at']
    
    # Almacenando los datos del usuario en un Dataframe
    df=[[tweet,id_user]] 
    df_tweet = pd.DataFrame(df)
    df_tweet.columns = ['tweet','user']
    
    # Aplicando la limpieza de los datos del usuario
    df_tweet_lim = limpieza(df_tweet)
    
    
    import joblib
    
    loaded_objects = joblib.load('tfidf_logistic_model.pkl')
    v = loaded_objects['tfidf_vectorizer']
    model = loaded_objects['logistic_model']
    
    df_tweet_tf = v.transform(df_tweet_lim['tweet'])
    pred = model.predict(df_tweet_tf)
    
 
    mapeo = {-1: "negativo", 0: "neutral", 1: "positivo"}
    vectorizer = np.vectorize(lambda x: mapeo.get(x, x))
    resultado = vectorizer(pred)
    
    #print("\n",resultado[0])
    #Obtenemos la base ded atos 
    base_musica = obtener_bdmusica()
    lista = listar_canciones(base_musica,resultado[0])
    lista.loc[:,["Album name","Track"]].to_csv("playlist.csv",index=False)
