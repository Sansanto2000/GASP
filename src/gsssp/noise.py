"""Ruido de escaneo y bordes de placa."""
from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

def add_background_field(
    img: NDArray[np.uint8],
    amplitude: float,
    grid: int = 8,
    *, rng: np.random.Generator = None
) -> NDArray[np.uint8]:
    """Aplica un campo de iluminacion de baja frecuencia, multiplicativo.

    Simula iluminacion despareja, vineteado optico y velo quimico desigual: efectos de
    baja frecuencia y bidireccionales que una placa fotografica escaneada real presenta
    y que el ruido de alta frecuencia de add_realistic_noise no cubre. Se aplica sobre
    el canvas antes de dibujar las observaciones, para que el gradiente las afecte
    tambien a ellas, como pasa fisicamente cuando la iluminacion del escaner es
    despareja.

    Se genera ruido en una grilla chica (grid x grid) y se escala al tamaño de la
    imagen con interpolacion bicubica, en vez de ruido por pixel, porque es lo que
    da la variacion de gran escala sin necesidad de otra dependencia.

    Parametros:
    - amplitude {float}: amplitud del campo. 0 no aplica ningun efecto (el canvas
    queda igual que antes). Valores mas altos oscurecen o aclaran zonas grandes de
    la placa de forma mas marcada.
    - grid {int}?: tamaño de la grilla base antes de interpolar. Default 8.
    - rng {np.random.Generator}?: generador aleatorio a usar. Si no se pasa se crea
    uno sin semilla.
    """
    if amplitude <= 0:
        return img

    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape[:2]
    small = rng.normal(1.0, amplitude, (grid, grid)).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)

def add_realistic_noise(
    img: NDArray[np.uint8],
    gaussian_std: float = 10.0,
    band_intensity: float = 5.0,
    speck_count: int = 10,
    speck_size: int = 3,
    blur_ksize: int = 3,
    violin_line_count: int = 0,
    violin_intensity = 0.7,
    violin_length_range = (0.05, 0.7),
    grain_std: float = 0.0,
    *, rng: np.random.Generator = None
) -> NDArray[np.uint8]:
    """Añadir ruido realista a una imagen.

    Trabaja en escala de grises: recibe y devuelve un arreglo (alto, ancho). La expansion
    a tres canales se hace una sola vez al final del pipeline, no en cada etapa.

    Parametros:
    - gaussian_std {float}?: ruido gaussiano. Simula imperfecciones naturales del sensor 
    o de la pelicula fotografica. Se basa en una distribucion normal o gaussiana. Default 10.0.
    - band_intensity {float}?: ruido de banda (horizontal o vertical). Default 5.0.
    - speck_count {int}?: cantidad de manchas de impuresa a simular.
    - speck_size {int}?: tamaño maximo de mancha de impuresa. Default 3.
    - blur_ksize {int}?: tamaño del kernel para desenfoque gaussiano. Debe ser impar, con 0 
    u otro valor invalido no se aplica ningun desenfoque. Default 3.
    - violin_line_count {int}?: cantidad de manchas alargadas tipo "violín" a simular. Default 0.
    - violin_intensity {float}?: intensidad de las manchas alargadas tipo "violín". Default 0.7.
    - violin_length_range {Tuple[float, float]}?: rango porcentual de longitud de las manchas
    alargadas tipo "violín". Default (0.05, 0.7).
    - grain_std {float}?: intensidad del grano de emulsion, ruido espacialmente
    correlacionado (a diferencia de gaussian_std, que es ruido blanco por pixel). 0 no
    aplica ningun efecto. Default 0.0.
    - rng {np.random.Generator}?: generador aleatorio a usar. Si no se pasa se crea uno
    sin semilla. Recibirlo permite que el resultado sea reproducible y seguro entre hilos.
    """
    if rng is None:
        rng = np.random.default_rng()

    img_noisy = img.astype(np.float32)

    # 1. Ruido gaussiano (general)
    noise = rng.normal(0, gaussian_std, img.shape)
    img_noisy += noise

    # 1b. Grano de emulsion: mismo ruido gaussiano pero desenfocado antes de sumarlo, para
    # que quede espacialmente correlacionado en vez de independiente por pixel.
    if grain_std > 0:
        grain = rng.normal(0, grain_std, img.shape).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (5, 5), 0)
        grain *= grain_std / (grain.std() + 1e-6)
        img_noisy += grain

    # 2. Ruido en bandas horizontales (tiras verticales o líneas horizontales)
    band = rng.normal(0, band_intensity, (img.shape[0], 1))
    img_noisy += band

    # 3. Puntos blancos o manchas (tipo polvo o defecto)
    for _ in range(speck_count):
        cx = rng.integers(0, img.shape[1])
        cy = rng.integers(0, img.shape[0])
        radius = 1 if speck_size <= 1 else rng.integers(1, speck_size)
        color = rng.integers(150, 255)  # blanco sucio
        cv2.circle(img_noisy, (int(cx), int(cy)), int(radius), int(color), cv2.FILLED)

    # 4. Manchas alargadas
    violin_sigma: float = 6.0
    h, w = img.shape[:2]
    # Ejes como columna y fila: al combinarlos broadcastean a (h, w) sin materializar
    # dos grillas completas por mancha, como hacia np.meshgrid.
    eje_y = np.arange(h)[:, None]
    eje_x = np.arange(w)[None, :]
    for _ in range(violin_line_count):
        violin_length_ratio: float = rng.uniform(*violin_length_range)

        # Centro x, y aleatorio
        y0 = rng.integers(0, h)
        # Centro horizontal (evita bordes)
        x0 = rng.integers(int(w * 0.1), int(w * 0.9)+1)

        # Largo horizontal (no llega a bordes)
        L = int(w * violin_length_ratio)
        sigma_x = L / 3

        # Gaussiana 2D estirada horizontalmente
        gaussian_2d = 255 * violin_intensity * np.exp(
            -(
                ((eje_y - y0) ** 2) / (2 * violin_sigma ** 2) +
                ((eje_x - x0) ** 2) / (2 * sigma_x ** 2)
            )
        )
        # Aplicar
        img_noisy += gaussian_2d

    # 5. Desenfoque suave (simula ópticas imperfectas)
    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        img_noisy = cv2.GaussianBlur(img_noisy, (blur_ksize, blur_ksize), 0)


    # Clip y convertir de vuelta a uint8
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    return img_noisy

class Position(Enum):
    RIGHT = 0
    LEFT = 1
    TOP = 2
    BOTTOM = 3

def add_plate_edge(img, edges, position:Position, *, rng:np.random.Generator = None):
    """Agrega un borde a la placa basado en los limites de las etiquetas.

    Trabaja en escala de grises: recibe y devuelve un arreglo (alto, ancho).

    Args:
        img (NDArray[np.uint8]): imagen a modificar.
        edges (tupla): (x_min, x_max, y_min, y_max) limites en pixeles donde 
        se mueven las etiquetas.
        position (Position): lado de la placa donde agregar el borde.
        rng (np.random.Generator, optional): generador aleatorio a usar. Si no se pasa
        se crea uno sin semilla.

    Returns:
        NDArray[np.uint8]: imagen con el borde agregado.
    """

    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape[:2]
    x_min, x_max, y_min, y_max = edges
    margin = 0.7
    # color del fondo "de atrás"
    gray = int(rng.integers(50, 156))
    bg_color = gray
    # color de la línea límite
    angle_noise = int(min(w, h) * 0.02)
    shift = int(rng.integers(-angle_noise, angle_noise + 1))

    match position:
        case Position.RIGHT:
            max_thickness = int((w - x_max) * (1 - margin))
            thickness = int(rng.integers(0, max(1, int(max_thickness)) + 1))
            pts = np.array([
                [w-thickness, 0],
                [w, 0],
                [w, h],
                [w-thickness + shift, h]
            ])
            cv2.fillPoly(img, [pts], bg_color)
        case Position.LEFT:
            max_thickness = int(x_min * (1 - margin))
            thickness = int(rng.integers(0, max(1, int(max_thickness)) + 1))
            pts = np.array([
                [0, 0],
                [thickness, 0],
                [thickness + shift, h],
                [0, h]
            ])
            cv2.fillPoly(img, [pts], bg_color)
        case Position.TOP:
            max_thickness = int(y_min * (1 - margin))
            thickness = int(rng.integers(0, max(1, int(max_thickness)) + 1))
            pts = np.array([
                [0, 0],
                [w, 0],
                [w, thickness],
                [0, thickness + shift]
            ])
            cv2.fillPoly(img, [pts], bg_color)
        case Position.BOTTOM:
            max_thickness = int((h - y_max) * (1 - margin))
            thickness = int(rng.integers(0, max(1, int(max_thickness)) + 1))
            pts = np.array([
                [0, h],
                [w, h],
                [w, h-thickness],
                [0, h-thickness + shift]
            ])
            cv2.fillPoly(img, [pts], bg_color)
    return img
