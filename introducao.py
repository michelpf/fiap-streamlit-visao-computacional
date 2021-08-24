import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import cv2
import io

"""
# Computer Vision

## Introdução 

Uma imagem pode ser representada de maneiras diferentes. A representação RGB, em duas dimensões, utiliza
 3 canais (ou matrizes) para as cores, na sequência, vermelho (red), verde (green) e azul (blue).
A imagem resultante é uma imagem colorida.

Analise a imagem a seguir ou envie outra imagem. Os comentários para cada canal de cor é referente a 
imagem original.
"""
imagem = cv2.imread("imagens/colorful-182220_1280.jpg")
st.image(imagem, channels="BGR")

"""
O shape da imagem é definido pela altura, comprimento e número de canais.
"""
imagem.shape

uploaded_file = st.file_uploader(
    'Tente uma outra imagem', type="png")
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    st.image(imagem, channels="BGR")
    imagem.shape

"""
## Visualização de imagem em matrizes 

Cada pixel da imagem representa um valor de 0 a 255 relacionado a intensidade de cor para cada canal, 
deste modo temos 256 possibilidades de vermelho, verde e azul além de suas combinações.

### Componente Vermelho

Quanto mais claro a imagem, mais intenso é a cor vermelha. Como na imagem original não há
a cor vermelha concentrada, a maioria das regiões é mais escura.
"""

b, g, r = cv2.split(imagem)

st.image(r)

r

"""
### Componente Verde

Na cor verde, percebemos que na metade de imagem, composta de verde é mais intensa, portanto 
temos imagem mais clara.
"""

st.image(g)

"""
### Componente Azul

A cor azul, tal qual a vermelha, não possui concentração, logo a maioria das regiões são escuras 
também.
"""

st.image(b)
