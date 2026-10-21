"""Redimensionado de placas a tamaño fijo y reubicacion de sus etiquetas."""
from enum import Enum

import cv2
import numpy as np

from gsssp.labels import LabelFormat


class ResizeMode(Enum):
    """Como se lleva la placa a un tamaño fijo.

    - STRETCH: cada eje se escala por separado hasta llenar el tamaño pedido. Usa la placa
    entera, pero la deforma: un rectangulo rotado pasa a ser un paralelogramo y la
    inclinacion aparente se amplifica en proporcion ancho/alto de la placa. Por eso solo
    admite etiquetas AABB: con OBB el generador lo rechaza (ver `validate_resize`).
    - LETTERBOX: un unico factor para ambos ejes, que conserva la proporcion, y relleno con
    el gris de fondo de la placa hasta completar el tamaño. Es lo que hace Ultralytics con
    imagenes reales. Cuesta resolucion en el eje corto de la placa: con una placa 2.5 veces
    mas ancha que alta, todo mide ~2.5 veces menos en vertical que con STRETCH.
    - NONE: no se redimensiona. Las placas conservan su tamaño original, asi que los lotes
    quedan de tamaño variable.
    """
    STRETCH = 0
    LETTERBOX = 1
    NONE = 2


# Afin identidad sobre coordenadas normalizadas: (sx, sy, ox, oy).
_IDENTIDAD = (1.0, 1.0, 0.0, 0.0)


def resize_plate(img, resize_shape, mode, *, fill=None):
    """Lleva una placa a `resize_shape` segun el modo y devuelve como se movieron sus ejes.

    Parametros:
    - img {NDArray[np.uint8]}: la placa, con forma (alto, ancho) o (alto, ancho, canales).
    - resize_shape {tuple[int, int]}: (ancho, alto) del tamaño final. Se ignora con NONE.
    - mode {ResizeMode}: como se redimensiona.
    - fill {int | None}: nivel de gris del relleno de LETTERBOX. Por defecto, la mediana de
    la placa ya redimensionada: la placa es en su mayor parte fondo, asi que la mediana es
    su gris de fondo, y calcularla sobre la version chica cuesta una fraccion de hacerlo
    a resolucion original. Se ignora en los otros modos.

    Return:
    - {NDArray[np.uint8]}: la placa redimensionada.
    - {tuple[float, float, float, float]}: `(sx, sy, ox, oy)`, la afin que lleva las
    coordenadas normalizadas de la placa a las de la imagen devuelta:
    `x' = x*sx + ox`, `y' = y*sy + oy`. Es lo que espera `remap_observation`.
    """
    if mode is ResizeMode.NONE:
        return img, _IDENTIDAD

    if resize_shape is None:
        raise ValueError(
            f"{mode.name} necesita resize_shape=(ancho, alto). "
            "Para no redimensionar se usa resize_mode=ResizeMode.NONE."
        )
    ancho_out, alto_out = int(resize_shape[0]), int(resize_shape[1])
    if mode is ResizeMode.STRETCH:
        # La placa llena todo el tamaño final, asi que las coordenadas normalizadas no se mueven.
        return cv2.resize(img, (ancho_out, alto_out)), _IDENTIDAD

    alto, ancho = img.shape[:2]
    # Un unico factor para ambos ejes: el que hace que la placa entre en el tamaño pedido.
    r = min(ancho_out / ancho, alto_out / alto)
    nuevo_ancho = min(ancho_out, max(1, round(ancho * r)))
    nuevo_alto = min(alto_out, max(1, round(alto * r)))
    contenido = cv2.resize(img, (nuevo_ancho, nuevo_alto))
    if fill is None:
        fill = int(round(float(np.median(contenido))))
    # Relleno entero y centrado, como el LetterBox(center=True) de Ultralytics. Entero para
    # que la placa caiga exactamente sobre la grilla de pixeles y la etiqueta no tenga
    # medio pixel de error.
    izq = (ancho_out - nuevo_ancho) // 2
    arr = (alto_out - nuevo_alto) // 2
    canvas = np.full((alto_out, ancho_out) + img.shape[2:], fill, dtype=img.dtype)
    canvas[arr:arr + nuevo_alto, izq:izq + nuevo_ancho] = contenido
    # La escala efectiva es nuevo/original y no `r`: el redondeo del tamaño hace que difieran
    # una fraccion de pixel, y la etiqueta tiene que seguir lo que cv2.resize hizo de verdad.
    return canvas, (
        nuevo_ancho / ancho_out, nuevo_alto / alto_out,
        izq / ancho_out, arr / alto_out,
    )


def remap_observation(label, affine):
    """Reubica una observacion y sus componentes en la imagen redimensionada.

    Las coordenadas siguen normalizadas, pero ahora respecto de la imagen devuelta por
    `resize_plate`. Con LETTERBOX eso mueve y achica las cajas (la placa ocupa solo una parte
    del lienzo); con STRETCH y NONE la placa llena el lienzo y las coordenadas no cambian.

    La misma afin sirve para la caja alineada a los ejes (`x_center_norm`, `width_norm`, ...)
    y para las esquinas (`corners_norm`). Con un escalado uniforme (LETTERBOX, NONE) los
    rectangulos siguen siendo rectangulos y los componentes siguen dentro de la observacion.
    Con STRETCH las esquinas quedarian como paralelogramos, que el formato OBB no representa:
    esa combinacion no se aproxima, se rechaza al construir el generador (`validate_resize`).

    Parametros:
    - label {dict}: observacion con `class_id`, `x_center_norm`, `y_center_norm`, `width_norm`,
    `height_norm`, `corners_norm` y `components`, como la produce `draw_observation`. No se
    modifica.
    - affine {tuple[float, float, float, float]}: la que devolvio `resize_plate`.

    Return:
    - {dict}: observacion nueva, con las mismas claves y sus `components` reubicados en el
    mismo orden.
    """
    sx, sy, ox, oy = affine
    nuevas = []
    for parte in [label] + list(label.get("components", [])):
        esquinas = np.asarray(parte["corners_norm"], dtype=np.float64).reshape(4, 2)
        esquinas = esquinas * (sx, sy) + (ox, oy)
        nuevas.append({
            "class_id": parte["class_id"],
            "x_center_norm": parte["x_center_norm"] * sx + ox,
            "y_center_norm": parte["y_center_norm"] * sy + oy,
            "width_norm": parte["width_norm"] * sx,
            "height_norm": parte["height_norm"] * sy,
            "corners_norm": [float(c) for c in esquinas.ravel()],
        })
    nueva = nuevas[0]
    nueva["components"] = nuevas[1:]
    return nueva


def validate_resize(resize_mode, resize_shape, label_format):
    """Rechaza al construir un generador las combinaciones de redimensionado que no sirven.

    Se rechaza en el constructor y no en el primer lote: una configuracion invalida no tiene
    que llegar a generar placas.

    - `resize_mode` tiene que ser un `ResizeMode`. Un texto o `None` caerian sin aviso en
    LETTERBOX, porque `resize_plate` trata como tal todo lo que no es STRETCH ni NONE.
    - STRETCH y LETTERBOX necesitan un `resize_shape` valido.
    - STRETCH no se admite con OBB. Escalar cada eje por separado convierte un rectangulo
    inclinado en un paralelogramo y el formato OBB solo representa rectangulos. Se rechaza
    siempre, haya o no inclinacion en un caso concreto, y no se aproxima con rectangulos.
    Con AABB no hay problema: la caja alineada a los ejes sigue siendo exacta.

    Parametros:
    - resize_mode {ResizeMode}, resize_shape {tuple[int, int] | None}, label_format
    {LabelFormat}: los del generador.

    Raises:
    - TypeError: `resize_mode` no es un `ResizeMode`.
    - ValueError: `resize_shape` invalido, o STRETCH con OBB.
    """
    if not isinstance(resize_mode, ResizeMode):
        raise TypeError(
            f"resize_mode debe ser un ResizeMode (STRETCH, LETTERBOX o NONE), se recibio {resize_mode!r}. "
            "Para no redimensionar se usa ResizeMode.NONE, no None ni un texto."
        )
    if resize_mode is ResizeMode.NONE:
        return
    if resize_shape is None or len(resize_shape) != 2 or min(resize_shape) <= 0:
        raise ValueError(
            f"{resize_mode.name} necesita resize_shape=(ancho, alto) con valores positivos, "
            f"se recibio {resize_shape!r}. Para no redimensionar se usa resize_mode=ResizeMode.NONE."
        )
    if resize_mode is ResizeMode.STRETCH and label_format is LabelFormat.OBB:
        raise ValueError(
            "ResizeMode.STRETCH con LabelFormat.OBB no esta permitido: escalar cada eje por separado "
            "convierte las cajas inclinadas en paralelogramos, y el formato OBB solo representa "
            "rectangulos. Alternativas: resize_mode=ResizeMode.LETTERBOX o ResizeMode.NONE, que "
            "conservan la geometria; o label_format=LabelFormat.AABB, que no se ve afectado."
        )


def validate_crop_resize(crop_resize_shape, label_format):
    """Rechaza al construir un generador de recortes el estirado con etiquetas OBB.

    Cada recorte se estira a `crop_resize_shape` sin tocar las etiquetas, asi que los
    componentes quedan como paralelogramos, que el formato OBB no representa. Igual que con
    STRETCH, se rechaza siempre y no se aproxima.

    Parametros:
    - crop_resize_shape {tuple[int, int] | None}, label_format {LabelFormat}: los del
    generador.

    Raises:
    - ValueError: OBB con un tamaño de recorte fijo.
    """
    if crop_resize_shape is not None and label_format is LabelFormat.OBB:
        raise ValueError(
            f"LabelFormat.OBB con crop_resize_shape={tuple(crop_resize_shape)} no esta permitido: el recorte "
            "se estira a ese tamaño y sus componentes quedan como paralelogramos, que el formato OBB no "
            "representa. Alternativas: crop_resize_shape=None, que conserva la geometria (los recortes "
            "salen en su tamaño original, como lista); o label_format=LabelFormat.AABB, que no se ve afectado."
        )
