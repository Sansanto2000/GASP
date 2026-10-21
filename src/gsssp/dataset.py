"""Volcado a disco de un dataset en formato YOLO."""
import os

import cv2
import numpy as np
from tqdm import tqdm

from gsssp.generators.spectrumLabeledSequence import SpectrumLabeledSequence
from gsssp.labels import (
    LabelClass,
    LabelFormat,
    label_list_to_yolov11_aabb_format,
    label_list_to_yolov11_obb_format,
)
from gsssp.resize import ResizeMode

# Formato y formateador de texto salen de la misma entrada: asi no pueden quedar
# desincronizados. Con el formateador AABB sobre filas OBB no hay error: toma los primeros 5
# valores y escribe una etiqueta de forma valida pero con otro significado, que Ultralytics
# acepta sin avisar.
_FORMATTERS = {
    LabelFormat.AABB: label_list_to_yolov11_aabb_format,
    LabelFormat.OBB: label_list_to_yolov11_obb_format,
}


def save_yolo_dataset(
    output_dir,
    *,
    batch_count=3000,
    batch_size=32,
    begin_num=0,
    seed=0,
    label_format=LabelFormat.AABB,
    label_classes=(LabelClass.OBSERVATION,),
    resize_mode=ResizeMode.NONE,
    height_range=(500, 1400),
    width_range=(500, 1400),
    progress=True,
    **generator_kwargs,
) -> int:
    """Genera placas y las vuelca a `output_dir` como un dataset en formato YOLO.

    Escribe `images/`, `labels/` y un `data.yaml` minimo. Los indices de clase son los
    canonicos de `LabelClass`, sin compactar, y el `data.yaml` declara siempre las tres
    clases: Ultralytics exige que todo indice de etiqueta sea menor que la cantidad de
    nombres, asi que con solo `science` y `lamp` (1 y 2) rechazaria el dataset. Sin `path`,
    Ultralytics toma la carpeta del yaml. `train` y `val` apuntan a las mismas imagenes: para
    validar de verdad, generar otro bloque con otra `seed` y apuntar `val` ahi.

    Parametros:
    - output_dir: carpeta de salida. Se crea si no existe.
    - batch_count: cantidad de lotes a generar. El total de imagenes es
    batch_count * batch_size.
    - batch_size: cantidad de imagenes por lote.
    - begin_num: numero desde el que arranca la numeracion de los archivos, y tambien el
    indice del primer lote del generador. Sirve para ampliar un dataset sin pisar lo que ya
    hay.
    - seed: semilla del generador. Con la misma semilla, el mismo begin_num y los mismos
    parametros se obtiene el mismo dataset.
    - label_format: `LabelFormat.AABB` u `LabelFormat.OBB`.
    - label_classes: clases a etiquetar, como tupla de `LabelClass`.
    - resize_mode: como se lleva cada placa a `resize_shape`. Default `ResizeMode.NONE`:
    cada placa sale a su tamaño original. `STRETCH` con OBB lo rechaza el generador.
    - height_range, width_range: rango del alto y el ancho de la placa en pixeles.
    - progress: mostrar la barra de progreso.
    - **generator_kwargs: el resto de los parametros de `SpectrumLabeledSequence`
    (`resize_shape`, rangos de ruido, `prob_edge`, etc.).

    Return:
    - {int}: cantidad de imagenes escritas.
    """
    # Falla ahora con un KeyError claro si el formato no tiene formateador.
    formatter = _FORMATTERS[label_format]

    sequence = SpectrumLabeledSequence(
        height_range=height_range,
        width_range=width_range,
        batch_size=batch_size,
        resize_mode=resize_mode,
        label_format=label_format,
        label_classes=label_classes,
        seed=seed,
        **generator_kwargs,
    )

    # Se crean antes del bucle: si falta alguna, cv2.imwrite devuelve False sin lanzar nada
    # y la corrida termina "bien" dejando etiquetas sin imagen.
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write("train: images\nval: images\nnames:\n")
        f.writelines(f"  {c.value}: {c.yolo_name}\n" for c in LabelClass)

    i = begin_num
    for _ in tqdm(range(batch_count), disable=not progress):
        batch_x, batch_y = sequence[i]

        for x, y in zip(batch_x, batch_y):
            filepath = os.path.join(images_dir, f"{i}.jpg")
            # Cortar en vez de avisar y seguir: si no se pudo escribir una, no se van a
            # poder escribir las que faltan, y continuar deja el dataset descalzado.
            if not cv2.imwrite(filepath, x):
                raise RuntimeError(
                    f"No se pudo guardar la imagen en {filepath}. Verificar ruta y permisos."
                )

            # Las filas de relleno del tensor ragged son todo ceros.
            y = y.numpy()
            filtered = y[~np.all(y == 0, axis=1)]

            # Siempre se escribe el archivo, aunque quede vacio: Yolo lo interpreta como
            # imagen de fondo, y mantener la correspondencia 1 a 1 con las imagenes permite
            # verificar que no se perdio nada contando archivos.
            with open(os.path.join(labels_dir, f"{i}.txt"), "w") as f:
                f.write("\n".join(map(formatter, filtered)))

            i += 1

    return i - begin_num
