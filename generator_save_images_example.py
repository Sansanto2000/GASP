'''
Docstring for generator_use_example
Se crea un lote de imagenes con el generador SpectrumLabeledSequence 
compatible con Tensorflow.
'''

import os
import sys
from pathlib import Path
import cv2
from dotenv import load_dotenv
from tqdm import tqdm
from gsssp.labels import label_list_to_yolov11_format
import numpy as np
from gsssp.generators.spectrumLabeledSequence import SpectrumLabeledSequence

### Configuracion de la corrida ###
# Las variables las lee este script, no la libreria: gsssp recibe todo por parametro.
# Se apunta al .env que esta al lado de este archivo, no al del directorio actual, para
# que la corrida no dependa de desde donde se invoque. Lo que ya este en el entorno tiene
# prioridad sobre el .env, asi se puede pisar un valor puntual sin editar el archivo.
load_dotenv(Path(__file__).with_name(".env"))

DESTINY = os.getenv("GASP_OUTPUT_DIR")
if not DESTINY:
    sys.exit(
        "Falta la variable GASP_OUTPUT_DIR.\n"
        "Copiar el archivo de ejemplo y completar la ruta de salida:\n"
        "    cp .env.example .env"
    )

BATCHT_SIZE = int(os.getenv("GASP_BATCH_SIZE", "32"))
BATCHT_CANT = int(os.getenv("GASP_BATCH_COUNT", "3000"))
BEGIN_NUM = int(os.getenv("GASP_BEGIN_NUM", "0"))

spectrum_gen = SpectrumLabeledSequence(
    height_range=(500,2000),
    width_range=(500,2000),
    batch_size=BATCHT_SIZE, 
    resize_shape=(640,640)
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
        lines = map(label_list_to_yolov11_format, filtered)
    
        # Guardar etiquetas
        filepath = os.path.join(DESTINY,"labels",f"{i}.txt")
        if lines:
            with open(filepath, "w") as f:
                f.write("\n".join(lines))

        # Incrementar contador
        i += 1
