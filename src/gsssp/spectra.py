"""Generacion de espectros sinteticos y curvas de desvanecimiento."""
import random
from typing import Callable
from enum import Enum

import numpy as np

class Fading(Enum):
    NO = 0
    GAUSSIAN = 1
    PLANCK = 2

def spectral_function(width:int, noise_level:float, n_peaks:int, baseline:int = 0, 
                      vertical_noise_level:float=0.2, peak_spread:float=1.0, 
                      n_absorption_lines:int=0, 
                      absorption_lines_spread:float = 1.0,
                      fading:Fading = Fading.PLANCK) -> Callable[[int], int]:
    """Genera una funcion que representa un espectro de ciencia sintetico.

    Parametros:
    - width {int}: ancho que tienen que cubrir los resultados.
    - noise_level {float}: Amplitud del ruido base (sobre 255).
    - n_peaks {int}: cantidad de picos a simular.
    - baseline {int}: valor minimo.
    - vertical_noise_level {float}?: Amplitud del ruido base vertical (sobre 255).
    Default 0.2.
    - peak_spread {float}?: multiplicador que afecta al ancho de los picos simulados. 
    Default 1.0.
    - n_absorption_lines {int}?: cantidad de lineas de absorción a simular. Default 0.
    - absorption_lines_spread {float}?: multiplicador que afecta al ancho de los las 
    lineas de absorcion simuladas. Default 1.0.

    Return:
    - {Callable[[int], int]}: funcion que dado un valor entero informa la intensidad
    que le corresponde.
    """

    x = np.arange(width)

    # Crear fondo con ruido blanco gaussiano centrado en 0
    noise = np.random.normal(loc=0.0, scale=noise_level, size=width)

    # Espectro inicial como ruido (más ruido positivo)
    spectrum = noise.clip(min=0)

    # Agregar n picos gaussianos con alturas y anchos aleatorios
    for _ in range(n_peaks):
        peak_center = np.random.uniform(0, width)
        peak_width = np.random.uniform(width*0.01, width*0.1) * peak_spread
        peak_height = np.random.uniform(50, 255)

        # Gaussiana: height * exp(- (x - center)^2 / (2*sigma^2))
        gaussian_peak = peak_height * np.exp(- (x - peak_center)**2 / (2 * peak_width**2))

        spectrum += gaussian_peak
    
    # Agregar n líneas de absorción (gaussianas invertidas)
    for _ in range(n_absorption_lines):
        abs_center = np.random.uniform(0, width)
        abs_width = np.random.uniform(width*0.01, width*0.04) * absorption_lines_spread
        abs_depth = np.random.uniform(20, 100)  # Qué tan profundas son

        gaussian_absorption = abs_depth * np.exp(- (x - abs_center)**2 / (2 * abs_width**2))
        if random.choice([0,1]) == 0:
            spectrum -= gaussian_absorption
        else:
            spectrum += gaussian_absorption # Algunas lineas las suma

    # Pequeñas lineas blancas aleatorias
    spectrum += np.random.rand(width)*vertical_noise_level

    ### Desvanecimiento
    match fading:
        case Fading.GAUSSIAN:
            x_idx = np.arange(width)
            center = width * 0.5
            sigma = width * 0.3  # Ancho
            gaussian_fade = np.exp(- (x_idx - center)**2 / (2 * sigma**2))
            spectrum *= gaussian_fade
        case Fading.PLANCK:
            #x_idx = np.arange(width)
            lambdas = np.linspace(0.2, 0.8, width)
            curve = planck_like(lambdas, T=0.5)
            planck_fade = (curve - np.min(curve)) / (np.max(curve) - np.min(curve)) # Normalizar
            spectrum *= planck_fade

    # Normalizar a rango [0, 1]
    spectrum -= spectrum.min()
    if spectrum.max() > 0:
        spectrum /= spectrum.max()

    # Escalar a [baseline, 255]
    spectrum = baseline + spectrum * (255 - baseline)

    spectrum = spectrum.astype(np.uint8)

    def intensity(xi):
        """Intensidad para un indice suelto o para un array de indices.

        Fuera del rango [0, width) devuelve 0, igual que antes. Aceptar arrays
        permite pintar una parte entera de una sola vez en vez de pixel por pixel.
        """
        idx = np.asarray(xi)
        dentro = (idx >= 0) & (idx < width)
        valores = np.where(dentro, spectrum[np.clip(idx, 0, width - 1)], 0)
        if idx.ndim == 0:
            return int(valores)
        return valores

    return intensity


def planck_like(l, T=0.5):
    """Funcion de planck simplificada basada en nanometros (eje x) y
    Temperatura. 

    Args:
        l (_type_): Vector de nanometros.
        T (float, optional): Temperatura. Defaults to 0.5.

    Returns:
        _type_: Vector de intensidades.
    """
    return 1 / (l**5 * (np.exp(1/(l*T)) - 1))
