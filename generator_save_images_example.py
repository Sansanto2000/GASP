'''
Docstring for generator_use_example
Se crea un lote de imagenes con el generador SpectrumLabeledSequence 
compatible con Tensorflow.
'''

import os
import cv2
from tqdm import tqdm
from gsssp.observationArtist import labelListToYolov11Format
import numpy as np
from gsssp.generators.spectrumLabeledSequence import SpectrumLabeledSequence

DESTINY = '/mnt/data3/sponte/datasets/conGSSSP.large.3' # "D:\\Datasets\\conGSSSP_v2"
BATCHT_SIZE = 32
BATCHT_CANT = 3000 # Total de 96000 imagenes
BEGIN_NUM = 0 # Numero de inicio

spectrum_gen = SpectrumLabeledSequence(
    height_range=(500,2000),
    width_range=(500,2000),
    batch_size=BATCHT_SIZE, 
    resize_shape=(640,640), 
    max_predictions=20
)

### Guardar elementos de la cantidad de lotes indicados ###
batch_cant = BATCHT_CANT
i = BEGIN_NUM
for batch_nro in tqdm(range(batch_cant)):
    batch_x, batch_y = spectrum_gen[i]

    """ Guardar cada imagen y sus etiquetas correspondientes """
    for x, y in zip(batch_x, batch_y):

        # Guardar imagen sintetica
        filepath = os.path.join(DESTINY,"images",f"{i}.jpg")
        success = cv2.imwrite(filepath, x)

        if not success:
            print("¡Error al guardar la imagen! Verifica la ruta y permisos.")

        # Convertir etiquetas a formato Yolov11
        y = y.numpy()
        filtered = y[~np.all(y == 0, axis=1)]
        lines = map(labelListToYolov11Format, filtered)
    
        # Guardar etiquetas
        filepath = os.path.join(DESTINY,"labels",f"{i}.txt")
        if lines:
            with open(filepath, "w") as f:
                f.write("\n".join(lines))

        # Incrementar contador
        i += 1
