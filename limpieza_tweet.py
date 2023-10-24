# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:02:00 2023

@author: jjmm08
"""
import re
import spacy
from unidecode import unidecode

nlp = spacy.load('es_core_news_sm')


def eliminar_enlaces(texto):
    patron_enlace = r'https?://\S+'
    return re.sub(patron_enlace, '', texto)

# Aplicar la función a la columna en el índice 7 (octava columna)

def eliminar_stop(texto):
    doc = nlp(texto)
    texto_sin_stopwords = [token.text for token in doc if not token.is_stop]
    return ' '.join(texto_sin_stopwords)

def eliminar_tildes(texto):
    texto_transformado = ''
    for caracter in texto:
        if (caracter == 'ñ') or (caracter == 'Ñ'):
            texto_transformado += 'ñ'  # Conservar la "ñ"
        else:
            texto_transformado += unidecode(caracter)
    return texto_transformado

def lemmatizar(texto):

    # eliminar stopwords y lematizar el texto
    doc = nlp(texto)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens).strip()

def limpiar_especiales(texto):
    
    emoticones = [":)", ":(", ":D", ";)", "<3", ":P"]

    for emote in emoticones:
        texto = texto.replace(emote, f" {emote} ")

    # Patrón para eliminar caracteres especiales, excluyendo los emoticones y ñ-Ñ
    patron = r'[^\w\s:()ñÑ]|: '
    
    # Reemplaza caracteres especiales por espacios en blanco
    texto_limpio = re.sub(patron, ' ', texto)

    return texto_limpio


def suprimir_repeticiones(texto):
    texto_suprimido = re.sub(r'(\S)\1{2,}', r'\1', texto)
    return texto_suprimido


def procesar_tweet(tweet):
    
    # Tokenizar el tweet
    doc = nlp(tweet)
    # Eliminar stopwords y lematizar
    tokens_procesados = [token.lemma_ for token in doc if not token.is_stop]
    # Reconstruir el tweet
    tweet_procesado = ' '.join(tokens_procesados)
    
    return tweet_procesado.strip()


def limpieza(df):
    data_tweet= df.copy()
    data_tweet.columns=['tweet','etiqueta']
    
    data_tweet['original'] = data_tweet['tweet']
    data_tweet['tweet'] = data_tweet['tweet'].apply(eliminar_enlaces)
    data_tweet['tweet'] = data_tweet['tweet'].str.replace(r'@\w+|#\w+', '', regex=True)
    data_tweet['tweet'] = data_tweet['tweet'].apply(suprimir_repeticiones)
    data_tweet['tweet'] = data_tweet['tweet'].str.replace(":)"," feliz ")
    data_tweet['tweet'] = data_tweet['tweet'].str.replace(":D"," feliz ")
    data_tweet['tweet'] = data_tweet['tweet'].str.replace(":-)"," feliz ") 
    data_tweet['tweet'] = data_tweet['tweet'].str.replace(":("," triste ")
    data_tweet['tweet'] = data_tweet['tweet'].apply(eliminar_tildes)
    data_tweet['tweet'] = data_tweet['tweet'].str.strip()
    data_tweet['tweet'] = data_tweet['tweet'].str.lower()
    data_tweet['tweet'] = data_tweet['tweet'].apply(limpiar_especiales)
    data_tweet['tweet'] = data_tweet['tweet'].str.replace('"', '')
    data_tweet['tweet'] = data_tweet['tweet'].str.replace(r'\d+', ' ', regex=True)
    data_tweet['tweet'] = data_tweet['tweet'].str.replace('\n', ' ')
    data_tweet['tweet'] = data_tweet['tweet'].str.replace(r'\s+', ' ', regex=True)
    data_tweet["tweet"] = data_tweet["tweet"].apply(procesar_tweet)
    data_tweet['tweet'] = data_tweet['tweet'].str.strip()
    return data_tweet  
