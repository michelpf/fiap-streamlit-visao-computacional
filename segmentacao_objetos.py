import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import cv2
import io
import random

from pathlib import Path
import gdown

import mrcnn.model as modellib
import mrcnn.visualize
import mrcnn.config

import os

cloud_location = "https://drive.google.com/uc?id=1pff606oZK08xhoe7pZi3SXi2g8mmDMyZ"

file_name = "mask_rcnn_coco.h5"


@st.cache
def download_weights():

    save_dest = Path('peso')
    save_dest.mkdir(exist_ok=True)

    f_checkpoint = Path("peso/" + file_name)

    if not f_checkpoint.exists():
        with st.spinner("Baixando pesos..."):
            gdown.download(cloud_location, "peso/" + file_name)


download_weights()


# Pesos pré-treinados
weight_file = 'peso/' + file_name
# Create model object in inference mode.

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"

    GPU_COUNT = 2
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 81


model = modellib.MaskRCNN(mode="inference",
                          config=SimpleConfig(),
                          model_dir="peso/")

model.load_weights(filepath=weight_file,
                   by_name=True)

file_names = next(os.walk("imagens/coco-dataset-2/"))[2]


"""
# Computer Vision

## Segmentação Avançada

Classificadores especializados em segmentação utilizando deep learning, como por exemplo a [Mask R CNN](https://arxiv.org/abs/1703.06870) combinam 
as propriedades dos classificadores de objetos para inicialmente identificar as instâncias de classes (retângulo delimitador) e então a partir desta região de interesse 
avaliar pixel a pixel qual a classe respectiva, segmentando desta forma a região de interesse muito mais preciso que os retângulos delimitadores.
"""
imagem = cv2.imread(os.path.join(
    "imagens/coco-dataset-2/", random.choice(file_names)))
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

st.image(imagem)

imagem.shape

uploaded_file = st.file_uploader(
    'Tente uma outra imagem', type=["png", "jpg"])
if uploaded_file is not None:
    img_stream = io.BytesIO(uploaded_file.getvalue())
    imagem = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    st.image(imagem)
    imagem.shape

"""
### Extraindo a região de interesse

O primeiro objetivo deste classificador é encontrar os objetos de interesse por meio da sua identificação por 
retângulos delimitadores. Assim o classificador pode focar sua atenção apenas nestas regiões de que realizar uma análise 
exaustiva na imagem por completo.
"""

r = model.detect([imagem], verbose=0)

r = r[0]

imagem_r = mrcnn.visualize.generate_display_instances(image=imagem,
                                                      boxes=r['rois'],
                                                      masks=r['masks'],
                                                      class_ids=r['class_ids'],
                                                      class_names=CLASS_NAMES,
                                                      scores=r['scores'])
st.image(imagem_r)

"""
### Segmentando as instâncias

Com as regiões de interesse separadas, o próximo passo é a análise pixel a pixel de cada uma das classes (ou instâncias) para que seja possível 
detectar um conjunto de pontos (ou um contorno) para que seja possível segmentar com precisão cada região de interesse. Assim é possível 
destactar suas instâncias basedas em cada tipo de classe. 
"""
imagem_r = mrcnn.visualize.generate_display_instances(image=imagem,
                                                      boxes=r['rois'],
                                                      masks=r['masks'],
                                                      class_ids=r['class_ids'],
                                                      class_names=CLASS_NAMES,
                                                      scores=r['scores'], only_boxes=False)
st.image(imagem_r)
