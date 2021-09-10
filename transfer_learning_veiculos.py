import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import cv2
import io

import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.applications import VGG16
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout

from pathlib import Path
import gdown

from keras.applications.vgg16 import preprocess_input

cloud_location = "https://drive.google.com/uc?id=15rb51Ekyc1EztNMteM9R-wToECtL5gta"


@st.cache
def download_model():

    save_dest = Path('modelo')
    save_dest.mkdir(exist_ok=True)

    f_checkpoint = Path("modelo/vehicle.h5")

    if not f_checkpoint.exists():
        with st.spinner("Baixando modelo..."):
            gdown.download(cloud_location, "modelo/vehicle.h5")


download_model()

vgg16 = VGG16(weights="imagenet", include_top=False,
              input_shape=(150, 150, 3))

model = vgg16.output
model = GlobalAveragePooling2D()(model)
model = Dropout(0.5)(model)
model = Dense(2, activation='softmax')(model)

model = Model(inputs=vgg16.input, outputs=model)

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=['accuracy'])

model.load_weights("modelo/vehicle.h5")


"""
# Computer Vision

# Transfer Learning

A técnica de transfer learning é utilizada amplamente em machine learning. Em visão computacional pode ser empregado
tanto em classificadores de imagens quanto nos de objetos.

Em síntese, o transfer learning transfere o aprendizado de um modelo que originalmente foi construído para uma determinada finalidade,
como por exemplo a detecção de imagens do dataset [ImageNet](https://www.image-net.org/) e permitir que o modelo seja
treinado com outro dataset, menor e mais simples. Sendo assim, a depender do caso, o modelo pode performar tão bem quanto
o original.

O modelo que vamos testar já foi treinado, o que iremos fazer é validar o aprendizado para a classificação
entre as 2 classes disponíveis: veículos ou não veículos.

Foi utilizado um dataset do [Kaggle](https://www.kaggle.com/brsdincer/vehicle-detection-image-set) e o treinamento foi realizado com base neste [Notebook](https://www.kaggle.com/taha07/vehicle-or-not-detection-100) de Taha.
"""
imagem = cv2.imread("imagens/bmw-918407_1280.jpg")

st.image(imagem, channels="BGR")


uploaded_file = st.file_uploader(
    'Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    st.image(imagem, channels="BGR")
    imagem.shape

"""
# Redimensionando a imagem

O classificador foi treinado com o tamanho das imagens 150 por 150. Desta forma precisamos de converter o tamanho adequado para utilizar 
o classificador.

O tamanho de imagem é herdado do classificador original.

"""

imagem_r = cv2.resize(imagem, (150, 150))
imagem_r = cv2.cvtColor(imagem_r, cv2.COLOR_BGR2RGB)

st.image(imagem_r)

"""
# Detecção de Veículo ou Não Veículo (Outros)

O classificador irá predizer sobre qual classe a imagem mais se aproxima e com qual probabilidade.
Note que o dataset foi composto de imagem de traseiras veículos na classe veículos e imagens de ruas e estradas 
considerando como não veículos. Portanto se as imagens de teste não seguirem a mesma forma poderá impactar 
na sua predição. 
"""

imagem_r.shape

image_p = imagem_r.reshape(
    (1, imagem_r.shape[0], imagem_r.shape[1], imagem_r.shape[2]))
image_p = preprocess_input(image_p)
pred_probs = model.predict(image_p)

# pred_probs

#resultado = ("{:.0f}% Veículo, {:.0f}% Outro".format(100*pred_probs[0, 1], 100*pred_probs[0, 0]))

if 100*pred_probs[0, 1] > 100*pred_probs[0, 0]:
    resultado = ("**É um veículo com {:.0f}% de probabilidade.**".format(
        100*pred_probs[0, 1]))
else:
    resultado = ("**Não é um veículo com {:.0f}% de probabilidade**".format(
        100*pred_probs[0, 0]))

resultado
