"""Redimensionado de placas a tamaño fijo y reubicacion de sus etiquetas."""
from enum import Enum

import cv2
import numpy as np


class ResizeMode(Enum):
    """Como se lleva la placa a un tamaño fijo.

    - STRETCH: cada eje se escala por separado hasta llenar el tamaño pedido. Usa la placa
    entera, pero la deforma: un rectangulo rotado pasa a ser un paralelogramo y la
    inclinacion aparente se amplifica en proporcion ancho/alto de la placa.
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


def _rectificar_grupo(poligonos, out_shape):
    """Reemplaza los paralelogramos de una observacion y de sus componentes por rectangulos
    que quedan anidados.

    El estirado es lineal, asi que los paralelogramos de los componentes estan exactamente
    dentro del de la observacion. Rectificar cada uno por separado lo rompe: el cizallado corre
    el centroide de cada componente a lo largo del ancho, y sus rectangulos dejan de calzar con
    el de la observacion (sobre 393 observaciones a 640x640, 309 tenian un componente
    sobresaliendo, hasta 31 px). Aca se usa un unico marco para todo el grupo, y las
    proyecciones de un conjunto contenido sobre un mismo eje quedan contenidas: las
    envolventes en ese marco se anidan por construccion, sin tolerancia.

    El marco es el del ancho de la observacion (`u`) y su perpendicular (`n`). Cada rectangulo
    es la envolvente de su paralelogramo en ese marco, asi que contiene al objeto. La de la
    observacion se acota para no salir del lienzo, y la de cada componente se recorta contra la
    de la observacion.

    Acotar es obligatorio: la envolvente de un paralelogramo inclinado puede salir de [0, 1]
    (sobre 256 placas a 640x640 lo hacia en 10, hasta 37 px) y Yolo descarta la placa entera
    cuando eso pasa. El acotado reduce el largo hacia el rectangulo inscrito, que esta dentro
    del paralelogramo y por lo tanto siempre cabe.

    Parametros:
    - poligonos {list[NDArray]}: cuadrilateros (x, y) normalizados, con forma (4, 2). El primero
    es la observacion y el resto sus componentes, con las esquinas en el orden de
    `cv2.boxPoints` (ver `_label_from_corners` en drawing.py): los lados (p0, p3) y (p1, p2) son
    los de ancho.
    - out_shape {tuple[int, int]}: (ancho, alto) en pixeles de la imagen a la que estan
    normalizados. Hace falta porque las perpendiculares solo valen en un espacio isotropo: en
    coordenadas normalizadas, con un tamaño no cuadrado, un pixel mide distinto en cada eje.

    Return:
    - {list[NDArray]}: un rectangulo normalizado, con forma (4, 2), por cada entrada y en el
    mismo orden, con las esquinas tambien en el orden de `cv2.boxPoints`.
    """
    escala = np.array(out_shape, dtype=np.float64)
    puntos = [p * escala for p in poligonos]
    obs = puntos[0]

    # Marco comun. El ancho sale de la estructura del orden de esquinas y no de cual par de
    # lados es el mas largo: tras estirar, una observacion alta es mas larga que ancha, y
    # esa heuristica daria un marco girado 90 grados respecto de sus propios componentes.
    # Las esquinas estan truncadas a entero, asi que se promedian los dos lados de ancho.
    vec_ancho = ((obs[3] - obs[0]) + (obs[2] - obs[1])) / 2
    largo = np.linalg.norm(vec_ancho)
    if largo < 1e-9:
        return poligonos   # caja degenerada: no hay direccion que alinear, y dividir daria NaN
    u = vec_ancho / largo
    n = np.array([-u[1], u[0]])

    def extension(p):
        pu, pn = p @ u, p @ n
        return pu.min(), pu.max(), pn.min(), pn.max()

    def rectangulo(u_min, u_max, n_min, n_max):
        # En el orden de cv2.boxPoints, el mismo de las etiquetas de entrada (n apunta hacia
        # "abajo"): abajo-izquierda, arriba-izquierda, arriba-derecha, abajo-derecha. Asi la
        # salida cumple el mismo contrato que la entrada.
        return np.array([
            u_min * u + n_max * n, u_min * u + n_min * n,
            u_max * u + n_min * n, u_max * u + n_max * n,
        ])

    # Observacion: envolvente en u, acotada por el lienzo. Cada esquina es
    # `punto medio del lado + o - semilargo * u`, asi que cada coordenada es lineal en el
    # semilargo y el tope sale de despejarlo en 0 <= coordenada <= tamaño.
    u_min, u_max, n_min, n_max = extension(obs)
    envolvente = u_max - u_min
    centro = (u_min + u_max) / 2
    tope = envolvente / 2
    cizallado = max(envolvente - largo, 0.0)   # cuanto se corren los lados de alto a lo largo de u
    piso = min(max(largo - cizallado, 0.0) / 2, tope)   # el rectangulo inscrito
    semi_largo = tope
    for n_borde in (n_min, n_max):
        medio = centro * u + n_borde * n
        for signo in (-1, 1):
            d = signo * u
            for k in (0, 1):
                if d[k] > 1e-12:
                    semi_largo = min(semi_largo, (escala[k] - medio[k]) / d[k])
                elif d[k] < -1e-12:
                    semi_largo = min(semi_largo, medio[k] / -d[k])
    semi_largo = max(semi_largo, piso)
    u_lo, u_hi = centro - semi_largo, centro + semi_largo

    rects = [rectangulo(u_lo, u_hi, n_min, n_max)]
    for p in puntos[1:]:
        a, b, c, d = extension(p)
        lo_u, hi_u = max(a, u_lo), min(b, u_hi)
        lo_n, hi_n = max(c, n_min), min(d, n_max)
        rects.append(rectangulo(lo_u, max(lo_u, hi_u), lo_n, max(lo_n, hi_n)))

    # El truncado a entero puede dejar una esquina del paralelogramo a una fraccion de pixel
    # del borde, y el rectangulo asoma por esa misma fraccion. Se recorta para no reabrir #31.
    return [np.clip(r, 0.0, escala) / escala for r in rects]


def remap_observation(label, affine, out_shape, mode):
    """Reubica una observacion y sus componentes en la imagen redimensionada.

    Las coordenadas siguen normalizadas, pero ahora respecto de la imagen devuelta por
    `resize_plate`. Con LETTERBOX eso mueve y achica las cajas (la placa ocupa solo una parte
    del lienzo); con STRETCH y NONE la placa llena el lienzo y las coordenadas no cambian.

    La caja alineada a los ejes (`x_center_norm`, `width_norm`, ...) se transforma con la
    afin y es exacta en los tres modos: un escalado por ejes mapea una caja alineada a los
    ejes en otra alineada, y lo anidado sigue anidado.

    Las esquinas (`corners_norm`) tambien pasan por la afin. En LETTERBOX y NONE el escalado
    es uniforme: los rectangulos siguen siendo rectangulos y los componentes siguen dentro de
    la observacion. En STRETCH no: el escalado anisotropo convierte cada rectangulo en un
    paralelogramo, que el formato OBB no puede representar y que Ultralytics reemplazaria por
    su cuenta con `cv2.minAreaRect`, etiqueta por etiqueta. Para que las etiquetas sean
    rectangulos validos y sigan anidadas, la observacion y sus componentes se rectifican
    juntos (ver `_rectificar_grupo`). Por eso esta funcion trabaja sobre la observacion entera
    y no sobre una etiqueta suelta, y por eso `SpectrumLabeledSequence` la aplica antes de
    elegir que clases etiquetar.

    Parametros:
    - label {dict}: observacion con `class_id`, `x_center_norm`, `y_center_norm`, `width_norm`,
    `height_norm`, `corners_norm` y `components`, como la produce `draw_observation`. No se
    modifica.
    - affine {tuple[float, float, float, float]}: la que devolvio `resize_plate`.
    - out_shape {tuple[int, int]}: (ancho, alto) en pixeles de la imagen devuelta.
    - mode {ResizeMode}: el modo con el que se redimensiono.

    Return:
    - {dict}: observacion nueva, con las mismas claves y sus `components` reubicados en el
    mismo orden.
    """
    sx, sy, ox, oy = affine
    partes = [label] + list(label.get("components", []))
    poligonos = [
        np.asarray(p["corners_norm"], dtype=np.float64).reshape(4, 2) * (sx, sy) + (ox, oy)
        for p in partes
    ]
    if mode is ResizeMode.STRETCH:
        poligonos = _rectificar_grupo(poligonos, out_shape)
    nuevas = [
        {
            "class_id": p["class_id"],
            "x_center_norm": p["x_center_norm"] * sx + ox,
            "y_center_norm": p["y_center_norm"] * sy + oy,
            "width_norm": p["width_norm"] * sx,
            "height_norm": p["height_norm"] * sy,
            "corners_norm": [float(c) for c in poligono.ravel()],
        }
        for p, poligono in zip(partes, poligonos)
    ]
    nueva = nuevas[0]
    nueva["components"] = nuevas[1:]
    return nueva
