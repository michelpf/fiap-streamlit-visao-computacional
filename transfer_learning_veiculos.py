import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import cv2
import io

import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout

from pathlib import Path
import gdown

from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

import os
import random


cloud_location = "https://drive.google.com/uc?id=1TVeH_TsmpftLg8xBkus50n0DkX27leiE"


def download_models():

    mobilenet_o = MobileNetV2(weights="imagenet")

    save_dest = Path('modelo')
    save_dest.mkdir(exist_ok=True)

    f_checkpoint = Path("modelo/vehicle_mobileNet.h5")

    if not f_checkpoint.exists():
        with st.spinner("Baixando modelo..."):
            gdown.download(cloud_location, "modelo/vehicle_mobileNet.h5")

    mobilenet_tl = MobileNetV2(weights="imagenet", include_top=False,
                               input_shape=(160, 160, 3))

    return mobilenet_o, mobilenet_tl


mobilenet_o, mobilenet_tl = download_models()


model = mobilenet_tl.output
model = GlobalAveragePooling2D()(model)
model = Dropout(0.5)(model)
model = Dense(2, activation='softmax')(model)

model = Model(inputs=mobilenet_tl.input, outputs=model)

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=['accuracy'])

model.load_weights("modelo/vehicle_mobileNet.h5")

"""
# Computer Vision

## Transfer Learning

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

file_names = next(os.walk("imagens/cars-non-cars/"))[2]
imagem = cv2.imread(os.path.join(
    "imagens/cars-non-cars/", random.choice(file_names)))
imagem = cv2.resize(imagem, (952, 1280))
#imagem = cv2.imread("imagens/bmw-918407_1280.jpg")

st.image(imagem, channels="BGR")

uploaded_file = st.file_uploader(
    'Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    st.image(imagem, channels="BGR")
    imagem.shape

"""
### Redimensionando a imagem para o modelo de transfer learning

O classificador foi treinado com a arquitetura e pesos do MobileNetV2 (os detalhes de implementação bem como o 
paper podem ser obtidos na [documentação](https://keras.io/api/applications/mobilenet/#mobilenetv2-function) do Keras) tamanho das imagens 160 por 160. Desta forma precisamos de converter o tamanho adequado para utilizar 
o classificador.

O tamanho de imagem é herdado do classificador original. Neste caso, apesar do classificador original requerer imagens do 
tamanho 224x224, o modelo por transfer learning pode admitir imagens de outras dimensões, que neste caso foi de 160x160.

"""

imagem_r = cv2.resize(imagem, (160, 160))
imagem_r = cv2.cvtColor(imagem_r, cv2.COLOR_BGR2RGB)

st.image(imagem_r)

"""
### Detecção de veículo ou não veículo (outros)

O classificador irá predizer sobre qual classe a imagem mais se aproxima e com qual probabilidade.
Note que o dataset foi composto de imagem de traseiras veículos na classe veículos e imagens de ruas e estradas 
considerando como não veículos. Portanto se as imagens de teste não seguirem a mesma forma (domínio) poderá impactar 
diretamente na sua predição. 
"""


imagem_r.shape

image_p = imagem_r.reshape(
    (1, imagem_r.shape[0], imagem_r.shape[1], imagem_r.shape[2]))

pred_probs = model.predict(image_p)

if 100*pred_probs[0, 1] > 100*pred_probs[0, 0]:
    resultado = ("**É um veículo com {:.0f}% de probabilidade.**".format(
        100*pred_probs[0, 1]))
else:
    resultado = ("**Não é um veículo com {:.0f}% de probabilidade**".format(
        100*pred_probs[0, 0]))

st.write(resultado)

"""
## Detecção das classes do ImageNet do modelo MobileNet

Caso optarmos por classificar a imagem com a arquitetura original do MobileNetV2 e os pesos do ImageNet, podemos obter 
suas classes originais.

Na implementação deste modelo será retornado as top 5 classes com maior probabilidade (em inglês).
"""

imagem_n = cv2.resize(imagem, (224, 224))
imagem_n = cv2.cvtColor(imagem_n, cv2.COLOR_BGR2RGB)

"""
### Redimensionamento da imagem do modelo original

O modelo original foi treinado com imagens de tamanho 224x224, por isso será esse o tamanho que precisaremos redimensionar 
a imagem de teste.

"""

st.image(imagem_n)

image_p = imagem_n.reshape(
    (1, imagem_n.shape[0], imagem_n.shape[1], imagem_n.shape[2]))
image_p = preprocess_input(image_p)

image_p.shape

preds = mobilenet_o.predict(image_p)

label_vgg = decode_predictions(preds)

"""
### Resultados da predição do modelo para as classes do ImageNet

Como este modelo não tem um domínio específico como foi o caso do modelo anterior por transfer learning, não há nenhuma 
limitação em relação a imagem de teste.

"""

resultado = ""

for prediction_id in range(len(label_vgg[0])):
    resultado = "Identificado **{} ** com {:.0f}% de probabilidade".format(
        label_vgg[0][prediction_id][1], label_vgg[0][prediction_id][2]*100)
    print(label_vgg[0][prediction_id])
    st.write(resultado)
