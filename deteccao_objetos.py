import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import cv2
import io

from pathlib import Path
import gdown

cloud_location = "https://drive.google.com/uc?id=1ZDP9eqLpLykZtPDgPzPiXgl_8l61VSgc"
cloud_location = "https://drive.google.com/uc?id=1QtpH_pvKrXXb8vQGnVZPp5N57OJtDifr"

file_name = "yolov3.weights"


@st.cache
def download_weights():

    save_dest = Path('peso')
    save_dest.mkdir(exist_ok=True)

    f_checkpoint = Path("peso/" + file_name)

    if not f_checkpoint.exists():
        with st.spinner("Baixando pesos..."):
            gdown.download(cloud_location, "peso/" + file_name)


download_weights()

# Configurações na rede neural YOLOv3
cfg_file = 'cfg/yolov3.cfg'
#m = Darknet(cfg_file)

# Pesos pré-treinados
weight_file = 'peso/' + file_name
# m.load_weights(weight_file)

# Rótulos de classes
names_file = 'data/coco.names'
#class_names = load_class_names(namesfile)

# Carregar os labels do conjunto de dados Coco
labels = open(names_file).read().strip().split("\n")

# Atribuir a cada label uma cor diferente (randômica)
np.random.seed(42)
cores = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# Carregar a rede
net = cv2.dnn.readNetFromDarknet(cfg_file, weight_file)


"""
# Computer Vision

## Detecção de Objetos

Classificadores do tipo de detecção de objetos são capazes de identificar, de forma separada, diferentes objetos em uma mesma imagem. 
Diferentemente dos classificadores de imagens, que deteccam uma classe por imagem, os classificadores de objetos por sua vez detectam várias classes (ou objetos) em uma 
mesma imagem.
Um destes classificadores que possui performance superior em tempo de detecção (50 ms) quando comparado com outros classificadores do mesmo tipo (250 ms), o [Yolo](https://pjreddie.com/darknet/yolo/), 
pode ser utilizado em aplicações em tempo real, utilizando hardware adequado.
"""
imagem = cv2.imread("imagens/baby-623417_1280.jpg")

st.image(imagem, channels="BGR")
imagem.shape

uploaded_file = st.file_uploader(
    'Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    imagem = cv2.resize(imagem, (952, 1280))
    st.image(imagem, channels="BGR")
    imagem.shape

"""
### Redimensionando a imagem para o tamanho do modelo

O classificador Yolo foi treinado com imagens de tamanho 608 x 608. Este modelo é mais leve e é indicado para hardwares com menos processamento.
Há outras variações do classificador para imagens de maior tamanho e poder computacional.

Note que como o resultado é um conjunto de coordenadas, podemos utilizar a imagem redimensionada para as detecções e logo depois 
interpolar os valores para desenhar na imagem original.
"""

imagem_r = cv2.resize(imagem, (608, 608))
imagem_r = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

st.image(imagem_r)

imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

"""
### Detecção de objetos

O classificador foi treinado com o dataset do [COCO](https://cocodataset.org/#home) (Common Objects in Context). O conjunto de imagens 
deste dataset representam 80 variadas classes de objetos do nosso cotidiano.

Por se tratar do modelo mais compacto (Yolo V3 320) dependendo da resolução nem todos os objetos serão facilmente detectados. Para detecções mais precisas utilize o classificador 
Yolo V3 416 ou 608.

Ainda há um outro classificador, o YoloV3 Tiny voltado para hardwares menos complexos com bom desempenho.
"""
imagem_o = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

conf_threshold = 0.1
nms_threshold = 0.1


def identificar_objetos(frame):

    (H, W) = frame.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Normalizando a imagem (fator de escala, tamanho, RGB/BGR)
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (608, 608),  swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Inicialização das bounding boxes
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            # Extração de pontuação e confiança
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filtrar somente o que for maior que o limiar de confiança
            if confidence > conf_threshold:

                # Definição do bounding box encontrado

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Aplicação do NMS (Non-Maxima Suppression) para eliminar overlapping de bounding boxes

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            # Após o filtro por NMS, desenhar os bounding boxes
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in cores[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(labels[classIDs[i]])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    return frame


imagem_f = identificar_objetos(imagem)
st.image(imagem_f)
