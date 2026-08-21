"""Dibujado de una observacion espectroscopica sobre una imagen."""
import random

import cv2
import numpy as np
from numpy.typing import NDArray

from gsssp.spectra import spectral_function

def _paint_part(canvas, ys, xs, spectrum_function, originX, baseGrey):
    """Pinta de una sola vez todos los pixeles de una parte de la observacion.

    Cada columna toma la intensidad que le corresponde segun el espectro, con
    baseGrey como piso para que la observacion se funda con el fondo.

    Parametros:
    - canvas {NDArray[np.uint8]}: imagen sobre la que se pinta (se modifica).
    - ys {NDArray}: coordenadas verticales de los pixeles de la parte.
    - xs {NDArray}: coordenadas horizontales de los pixeles de la parte.
    - spectrum_function {Callable}: funcion de intensidad devuelta por spectral_function.
    - originX {int}: coordenada X donde arranca el espectro.
    - baseGrey {int}: nivel de gris minimo a considerar.
    """
    if len(xs) == 0:
        return
    intensities = spectrum_function(xs - originX)
    intensities = np.maximum(intensities, baseGrey).astype(np.uint8)
    canvas[ys, xs] = intensities[:, None]

def drawObservation(
        img: NDArray[np.uint8], 
        x:int, y:int, width:int, height:int, 
        opening:float, distanceBetweenParts:float,
        angle:int=0, inplace:bool=True, 
        baseGrey:int = 1, debug:bool=True) -> NDArray[np.uint8]:
    """Funcion que recibe la informacion de una imagen en formato
    matricial y dibuja una observacion en la misma acorde a las 
    cordenadas especificadas.
    IMPORTANTE: a menos que se especifique la funcion modifica la 
    matriz recibida en vez de hacer una copia.
    IMPORTANTE: se espera que la imagen base sea oscura, para pintar
    la observacion se sigue una estrategia de pixel mas alto, si el 
    fondo en una parte que se superpone con la observacion tiene un
    color mas claro entonces se pinta el pixel del fondo. Esto es 
    para que la observacion pintada se mezcle bien con la imagen.

    params:
    - img {NDArray[np.uint8]}: matriz de pixeles que representa la 
    imagen.
    - x {int}: coordenada horizontal central donde se dibujara el espectro.
    - y {int}: coordenada vertical central donde se dibujara el espectro.
    - width {int}: ancho de la observación a dibujar.
    - height {int}: alto de la observación a dibujar.
    - openingLamp {float}: apertura porcentual de la lampara. (0, 0.5) acorde 
    al alto de la observacion.
    - distanceBetweenParts {float}: distancia porcentual entre partes de un 
    espectro. (0, 0.33) acorde al alto de la observacion.
    - angle {int}?: angulo de inclinacion de la observacion a dibujar.
    - inplace {bool}?: condicion que indica si se realizaran los cambios
    sobre la imagen recibida o sobre una copia. Default True.
    - baseGrey {int}?: nivel de gris minimo a considerar. Default 1.
    - debug {bool}?: al activar se pinta las cajas delimitadoras de la observacion
    generada sobre la imagen. Default False.

    return 
    - {NDArray[np.uint8]}: matriz de pixeles que representa la 
    imagen con la observacion agregada.
    - {NDArray[np.uint8]}: matriz de pixeles que representa solo la observacion
    generada acorde a las dimensiones de la imagen recibida. Todos los pixeles que
    no corresponden a la observación son 0.
    - {NDArray[np.uint8]}: mascara de locación del espectro.
    - {dict[str, Number]}: informacion de caja delimitadora de la observación (formato
    yolov11).
    """
    
    if not inplace:
        img = img.copy()


    # Crear el rectángulo de observacion rotado
    rectObservation = ((x, y), (width, height), angle)

    # Crear el rectangulo de cada parte de la observacións
    openingInPixel = round(height * opening)
    distanceBetweenPartsInPixel = round(height * distanceBetweenParts)
    centerLamp1 = (x, y-(height-openingInPixel)/2) 
    centerLamp2 = (x, y+(height-openingInPixel)/2)
    rectParts = {
        "lamp1": (centerLamp1, (width, openingInPixel), 0),
        "lamp2": (centerLamp2, (width, openingInPixel), 0),
        "science": ((x,y), (width, height-openingInPixel*2-distanceBetweenPartsInPixel*2), 0),
    }
    boxParts = {
        "lamp1": np.int32(cv2.boxPoints(rectParts["lamp1"])),
        "lamp2": np.int32(cv2.boxPoints(rectParts["lamp2"])),
        "science": np.int32(cv2.boxPoints(rectParts["science"])),
    }
    
    # Obtener las esquinas del rectángulo de observacion
    boxObservation = cv2.boxPoints(rectObservation)
    boxObservation = np.int32(boxObservation)

    # Preparar etiqueta de caja delimitadora para grafico
    allBoxs = np.concatenate([boxObservation], axis=0)
    x_coords = allBoxs[:,0]
    y_coords = allBoxs[:,1]
    labelForGraph: dict[str, Number] = {
        "x":x_coords.min(), 
        "y":y_coords.min(), 
        "width":x_coords.max() - x_coords.min(), 
        "height":y_coords.max() - y_coords.min(), 
    }

    # Preparar etiquetas en formato yolov11 para reportar al usuario
    img_height, img_width = img.shape[:2]
    labelObservation: dict[str, Number] = {
        "class_id": 0,
        "x_center_norm":((x_coords.min() + x_coords.max())/2)/img_width,
        "y_center_norm":((y_coords.min() + y_coords.max())/2)/img_height,
        "width_norm": labelForGraph["width"] / img_width,
        "height_norm": labelForGraph["height"] / img_height,
    }

    if (debug):
        cv2.rectangle(   # Etiqueta (Caja delimitadora)
            img,
            (labelForGraph["x"], labelForGraph["y"]),
            (labelForGraph["x"] + labelForGraph["width"], labelForGraph["y"] + labelForGraph["height"]), 
            (255,0,0), thickness=3
        )

    # Mascara para cada parte del espectro.
    maskParts = {
        "lamp1": np.zeros(img.shape[:2], dtype=np.uint8),
        "lamp2": np.zeros(img.shape[:2], dtype=np.uint8),
        "science": np.zeros(img.shape[:2], dtype=np.uint8) 
    }
    cv2.drawContours(maskParts["lamp1"], [boxParts["lamp1"]], 0, 255, thickness=cv2.FILLED)
    cv2.drawContours(maskParts["lamp2"], [boxParts["lamp2"]], 0, 255, thickness=cv2.FILLED)
    cv2.drawContours(maskParts["science"], [boxParts["science"]], 0, 255, thickness=cv2.FILLED)

    # Mascara general: la union de las partes, que ya estan dibujadas. Rasterizar los
    # mismos tres poligonos de nuevo daria identico resultado pero cuesta el triple.
    maskObservation = maskParts["lamp1"] | maskParts["lamp2"] | maskParts["science"]

    # Dominio horizontal de los espectros, en el sistema SIN rotar.
    # Las partes se dibujan sin inclinacion (la rotacion se aplica recien al final con
    # warpAffine), asi que el espectro tiene que medirse e indexarse en ese mismo sistema.
    # Usar el bounding box rotado (labelForGraph) desalinea el desvanecimiento y descarta
    # picos, y el desvio crece con el angulo.
    # El +1 es porque los extremos son inclusivos: drawContours pinta tambien la columna
    # partsX.max(), y sin el la ultima columna cae fuera del espectro y queda en baseGrey.
    partsX = np.concatenate([boxParts["lamp1"], boxParts["lamp2"], boxParts["science"]])[:,0]
    partsOriginX = int(partsX.min())
    partsWidth = int(partsX.max() - partsX.min()) + 1

    # Pintar espectro de ciencia
    onlyObservation = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    ys, xs = np.where(maskParts["science"] == 255)
    vertical_noise_level = random.uniform(0,0.05)
    science_function = spectral_function(
        width=partsWidth,
        noise_level=255*random.uniform(0, 0.01), 
        n_peaks=random.randint(4, 10),
        baseline=random.randint(max(0, baseGrey-60), baseGrey+15),
        vertical_noise_level= vertical_noise_level,
        peak_spread=random.uniform(0.4, 2.6),
        n_absorption_lines=random.randint(0, 12),
        absorption_lines_spread=random.uniform(0, 0.1),
        )
    _paint_part(onlyObservation, ys, xs, science_function, partsOriginX, baseGrey)

    # Pintar lampara de comparación 1
    lamp_function = spectral_function(
        width=partsWidth,
        noise_level=255*0.01, 
        n_peaks=random.randint(15, 150),
        baseline=random.randint(max(0, baseGrey-60), baseGrey+5),
        vertical_noise_level=vertical_noise_level,
        peak_spread=random.uniform(0.001, 0.04),
        n_absorption_lines=0,
        )
    ys, xs = np.where(maskParts["lamp1"] == 255)
    _paint_part(onlyObservation, ys, xs, lamp_function, partsOriginX, baseGrey)

    # Pintar lampara de comparación 2
    ys, xs = np.where(maskParts["lamp2"] == 255)
    _paint_part(onlyObservation, ys, xs, lamp_function, partsOriginX, baseGrey)

    # Rotar espectro y mascara acorde a la cantidad de grados.
    M = cv2.getRotationMatrix2D((x,y), angle, 1)
    onlyObservation = cv2.warpAffine(onlyObservation, M, (img.shape[1], img.shape[0]))
    maskObservation = cv2.warpAffine(maskObservation, M, (img.shape[1], img.shape[0]))

    # Fusionar con la imagen recibida
    # mask = maskObservation > 0  # shape: (H, W), dtype: bool
    # mask_3ch = np.stack([mask] * 3, axis=-1)  # shape: (H, W, 3)
    # img = np.where(mask_3ch, onlyObservation, img)    # Pintar observacion arriba
    img = np.maximum(img, onlyObservation)  # Quedarse con los pixeles mas altos.

    return img, onlyObservation, maskObservation, labelObservation
