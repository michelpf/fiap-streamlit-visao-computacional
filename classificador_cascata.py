import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import cv2
import io

"""
# Computer Vision

## Classificador de Cascata de Haar

Os classificadores em cascata ajudam nas tarefas de separação de região de interesse, isto é, 
etapa intermediária de separar no que focar dentro de uma imagem e assim utilizar classificadores mais
complexos.

Um exemplo comum é a detecção de faces em uma imagem originada de uma câmera da qual podem 
haver inúmeros elementos. Neste caso o classificador identifica as regiões de interesse, ou seja,
as faces na imagem. Posteriormente cada segmento é analizado individualmente, se necessário para 
realizar sua identificação.
"""
imagem = cv2.imread("imagens/employees-885338_1280.jpg")
st.image(imagem, channels="BGR")


uploaded_file = st.file_uploader(
    'Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    st.image(imagem, channels="BGR")
    imagem.shape

"""
## Simplificando a representação de imagem

A detecção de rostos (ou qualuer outro objeto) utilizando o classificador em cascata de Haar 
não necessita da imagem colorida, mas sim trabalha com a representação em escala de cinza,
tornando seus atributos mais simples de serem processados (ao invés de 3 canais, apenas 1).
"""

imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

st.image(imagem_gray)

"""
### Classificador

O resultado do classificador corresponde a uma ou mais regiões detectadas. Essas regiões estão no formato 
de um retângulo que delimita sua identificação, com isso podemos recortar a imagem baseada nestas 
coordenadas.
"""


classificador_face = cv2.CascadeClassifier(
    'classificadores/haarcascade_frontalface_default.xml')
faces = classificador_face.detectMultiScale(imagem_gray, 1.3, 4)

"""
Faces encontradas:
"""

len(faces)

"""
Coordenadas (x, y, comprimento, altura):
"""

faces

"""
Destaque das regiões de interesse encontradas (rostos):
"""
imagem_anot = imagem.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(imagem_anot, (x, y), (x+w, y+h), (0, 0, 255), 2)

st.image(imagem_anot, channels="BGR")

"""
### Regiões de interesse separadas

Com as coordenadas é possível recortar as imagens tendo como base a sua representação original 
em 3 canais (colorido), pois na identificação de rostos a cor pode ser um atributo relevante dada 
a complexidade de alguns modelos que podem empregados.

As imagens foram redimensionadas para facilitar a visão.

"""

col1, col2, col3 = st.columns([1, 1, 1])

for idx, (x, y, w, h) in enumerate(faces):
    caption = "Face " + str(idx+1) + ", localizada em (" + str(x) + \
        "," + str(y) + ") com dimensões " + str(w) + " x " + str(h)
    col2.image(imagem[y:y+h, x:x+w], channels="BGR",
               width=200, caption=caption,  use_column_width=True)
