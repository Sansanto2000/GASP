"""Ruido de escaneo y bordes de placa."""
import random
from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

def add_realistic_noise(
    img: NDArray[np.uint8],
    gaussian_std: float = 10.0,
    band_intensity: float = 5.0,
    speck_count: int = 10,
    speck_size: int = 3,
    blur_ksize: int = 3,
    violin_line_count: int = 0,
    violin_intensity = 0.7,
    violin_length_range = (0.05, 0.7)
) -> NDArray[np.uint8]:
    """Añadir ruido realista a una imagen.

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
    """
    img_noisy = img.astype(np.float32)

    # 1. Ruido gaussiano (general)
    noise = np.random.normal(0, gaussian_std, img.shape)
    img_noisy += noise

    # 2. Ruido en bandas horizontales (tiras verticales o líneas horizontales)
    band = np.random.normal(0, band_intensity, (img.shape[0], 1, 1))
    img_noisy += band

    # 3. Puntos blancos o manchas (tipo polvo o defecto)
    for _ in range(speck_count):
        cx = np.random.randint(0, img.shape[1])
        cy = np.random.randint(0, img.shape[0])
        radius = 1 if speck_size <= 1 else np.random.randint(1, speck_size)
        color = np.random.randint(150, 255)  # blanco sucio
        cv2.circle(img_noisy, (cx, cy), radius, (color,) * 3, cv2.FILLED)

    # 4. Manchas alargadas
    violin_sigma: float = 6.0
    h, w = img.shape[:2]
    for _ in range(violin_line_count):
        violin_length_ratio: float = np.random.uniform(*violin_length_range)

        # Centro x, y aleatorio
        y0 = np.random.randint(0, h)
        # Centro horizontal (evita bordes)
        x0 = np.random.randint(int(w * 0.1), int(w * 0.9)+1)

        # Largo horizontal (no llega a bordes)
        L = int(w * violin_length_ratio)
        sigma_x = L / 3

        # Grilla de coordenadas
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        # Gaussiana 2D estirada horizontalmente
        gaussian_2d = 255 * violin_intensity * np.exp(
            -(
                ((yy - y0) ** 2) / (2 * violin_sigma ** 2) +
                ((xx - x0) ** 2) / (2 * sigma_x ** 2)
            )
        )
        # Aplicar
        img_noisy += np.stack([gaussian_2d]*3, axis=-1)

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

def add_plate_edge(img, edges, position:Position):
    """Agrega un borde a la placa basado en los limites de las etiquetas.

    Args:
        img (NDArray[np.uint8]): imagen a modificar.
        edges (tupla): (x_min, x_max, y_min, y_max) limites en pixeles donde 
        se mueven las etiquetas.
        color (str): color del borde a agregar.

    Returns:
        NDArray[np.uint8]: imagen con el borde agregado.
    """

    h, w = img.shape[:2]
    x_min, x_max, y_min, y_max = edges
    margin = 0.7
    # color del fondo "de atrás"
    gray = random.randint(50, 155)
    bg_color = (gray, gray, gray)
    # color de la línea límite
    angle_noise = int(min(w, h) * 0.02)
    shift = random.randint(-angle_noise, angle_noise)

    match position:
        case Position.RIGHT:
            max_thickness = int((w - x_max) * (1 - margin))
            thickness = random.randint(0, max(1, int(max_thickness)))
            pts = np.array([
                [w-thickness, 0],
                [w, 0],
                [w, h],
                [w-thickness + shift, h]
            ])
            cv2.fillPoly(img, [pts], bg_color)
        case Position.LEFT:
            max_thickness = int(x_min * (1 - margin))
            thickness = random.randint(0, max(1, int(max_thickness)))
            pts = np.array([
                [0, 0],
                [thickness, 0],
                [thickness + shift, h],
                [0, h]
            ])
            cv2.fillPoly(img, [pts], bg_color)
        case Position.TOP:
            max_thickness = int(y_min * (1 - margin))
            thickness = random.randint(0, max(1, int(max_thickness)))
            pts = np.array([
                [0, 0],
                [w, 0],
                [w, thickness],
                [0, thickness + shift]
            ])
            cv2.fillPoly(img, [pts], bg_color)
        case Position.BOTTOM:
            max_thickness = int((h - y_max) * (1 - margin))
            thickness = random.randint(0, max(1, int(max_thickness)))
            pts = np.array([
                [0, h],
                [w, h],
                [w, h-thickness],
                [0, h-thickness + shift]
            ])
            cv2.fillPoly(img, [pts], bg_color)
    return img
