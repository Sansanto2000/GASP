"""Generacion de espectros (sinteticos o a partir de LAMOST) y curvas de desvanecimiento."""
from typing import Callable
from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

from gsssp.lamost import sample_lamost_window

class Fading(Enum):
    NO = 0
    GAUSSIAN = 1
    PLANCK = 2
    # Misma subida que PLANCK a la izquierda, pero despues del pico baja hasta 0 en vez
    # de quedarse en ~0.25: se desvanecen los dos extremos.
    PLANCK_BOTH_ENDS = 3

def spectral_function(width:int, noise_level:float, n_peaks:int, baseline:int = 0, 
                      vertical_noise_level:float=0.2, peak_spread:float=1.0, 
                      n_absorption_lines:int=0, 
                      absorption_lines_spread:float = 1.0,
                      fading:Fading = Fading.PLANCK,
                      *, rng:np.random.Generator = None) -> Callable[[int], int]:
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
    - fading {Fading}?: tipo de desvanecimiento a aplicar. Default Fading.PLANCK.
    - rng {np.random.Generator}?: generador aleatorio a usar. Si no se pasa se crea uno
    sin semilla. Recibirlo permite que el resultado sea reproducible y seguro entre hilos.

    Return:
    - {Callable[[int], int]}: funcion que dado un valor entero informa la intensidad
    que le corresponde.
    """

    if rng is None:
        rng = np.random.default_rng()

    x = np.arange(width)

    # Crear fondo con ruido blanco gaussiano centrado en 0
    noise = rng.normal(loc=0.0, scale=noise_level, size=width)

    # Espectro inicial como ruido (más ruido positivo)
    spectrum = noise.clip(min=0)

    # Agregar n picos gaussianos con alturas y anchos aleatorios
    for _ in range(n_peaks):
        peak_center = rng.uniform(0, width)
        peak_width = rng.uniform(width*0.01, width*0.1) * peak_spread
        peak_height = rng.uniform(50, 255)

        # Gaussiana: height * exp(- (x - center)^2 / (2*sigma^2))
        gaussian_peak = peak_height * np.exp(- (x - peak_center)**2 / (2 * peak_width**2))

        spectrum += gaussian_peak
    
    # Agregar n líneas de absorción (gaussianas invertidas)
    for _ in range(n_absorption_lines):
        abs_center = rng.uniform(0, width)
        abs_width = rng.uniform(width*0.01, width*0.04) * absorption_lines_spread
        abs_depth = rng.uniform(20, 100)  # Qué tan profundas son

        gaussian_absorption = abs_depth * np.exp(- (x - abs_center)**2 / (2 * abs_width**2))
        if rng.integers(0, 2) == 0:
            spectrum -= gaussian_absorption
        else:
            spectrum += gaussian_absorption # Algunas lineas las suma

    # Pequeñas lineas blancas aleatorias
    spectrum += rng.random(width)*vertical_noise_level

    ### Desvanecimiento
    spectrum *= _fading_curve(width, fading)

    # Normalizar a rango [0, 1]
    spectrum -= spectrum.min()
    if spectrum.max() > 0:
        spectrum /= spectrum.max()

    # Escalar a [baseline, 255]
    spectrum = baseline + spectrum * (255 - baseline)

    spectrum = spectrum.astype(np.uint8)

    return _intensity_function(spectrum)


def lamost_spectral_function(width:int, baseline:int = 0,
                             window_range=(3000, 6000),
                             smoothing_range=(1.0, 3.0),
                             fading:Fading = Fading.PLANCK_BOTH_ENDS,
                             exposure_range=(0.5, 1.0),
                             noise_range=(0.01, 0.01),
                             prob_flip=0.2,
                             *, rng:np.random.Generator = None) -> Callable[[int], int]:
    """Genera una funcion que representa un espectro de ciencia a partir de uno real.

    Toma un tramo de un espectro estelar de LAMOST (ver `gsssp.lamost`), a veces lo
    invierte (el sentido azul-rojo depende del espectrografo), lo suaviza, lo
    remuestrea al ancho pedido, lo desvanece hacia los extremos, le suma ruido por columna
    y le aplica una exposicion.

    El ruido se suma ya normalizado, como fraccion del tope, y atenuado por la misma curva
    de desvanecimiento. Sumarlo en las unidades del dataset, antes de normalizar, lo hacia
    dominante en los tramos de poco flujo (el azul de una estrella fria), que la
    normalizacion despues estiraba: la banda entera quedaba rayada.

    El flujo cero va a `baseline` y el percentil 99.5 del tramo a 255 (lo que lo supera
    satura). No se estira el minimo del tramo a `baseline`, como hace spectral_function:
    un tramo de continuo casi plano quedaria con su ruido pixel a pixel amplificado a
    todo el rango de grises, y la profundidad de las lineas dejaria de ser la real.

    Parametros:
    - width {int}: ancho que tienen que cubrir los resultados.
    - baseline {int}?: valor minimo. Default 0.
    - window_range {tuple[float, float]}?: rango del ancho del tramo, en angstrom. Ver
    sample_lamost_window. Default (3000, 6000).
    - smoothing_range {tuple[float, float]}?: rango del desvio del suavizado gaussiano,
    en pixeles de la grilla de LAMOST. Representa la resolucion del espectrografo de la
    placa, menor que la de LAMOST, y quita el ruido pixel a pixel del espectro original.
    Default (1.0, 3.0).
    - fading {Fading}?: tipo de desvanecimiento a aplicar. Default
    Fading.PLANCK_BOTH_ENDS: con flujo cero anclado al gris base, la cola de ~0.25 de
    PLANCK no llega a verse como desvanecimiento en el extremo derecho.
    - exposure_range {tuple[float, float]}?: rango del factor de exposicion, que escala
    el espectro ya normalizado. 1 lleva el tope a 255; menos de 1 simula una placa
    subexpuesta y baja el tope a baseline + factor * (255 - baseline). Default (0.5, 1.0).
    - noise_range {tuple[float, float]}?: rango del desvio del ruido gaussiano por
    columna, como fraccion del tope (0.01 = 1 %). Default (0.01, 0.01).
    - prob_flip {float}?: probabilidad de invertir el tramo, de rojo a azul. Default 0.2.
    - rng {np.random.Generator}?: generador aleatorio a usar. Si no se pasa se crea uno
    sin semilla. Recibirlo permite que el resultado sea reproducible y seguro entre hilos.

    Return:
    - {Callable[[int], int]}: funcion que dado un valor entero informa la intensidad
    que le corresponde.
    """
    if rng is None:
        rng = np.random.default_rng()

    tramo = sample_lamost_window(window_range, rng=rng).astype(np.float32)
    if rng.random() < prob_flip:
        tramo = tramo[::-1]
    sigma = rng.uniform(*smoothing_range)
    tramo = cv2.GaussianBlur(tramo[None, :], (0, 0), sigmaX=sigma,
                             borderType=cv2.BORDER_REFLECT)[0]

    # Al achicar se promedia por area: interpolar descartaria pixeles y haria aliasing
    # con las lineas finas. Al agrandar alcanza con interpolar.
    interpolacion = cv2.INTER_AREA if len(tramo) > width else cv2.INTER_LINEAR
    spectrum = cv2.resize(tramo[None, :], (width, 1), interpolation=interpolacion)[0]
    spectrum = spectrum.astype(np.float64)

    # Se aplica despues de invertir: la curva no es simetrica, y asi el lado que se
    # desvanece del todo es siempre el mismo, igual que en spectral_function.
    curve = _fading_curve(width, fading)
    spectrum *= curve

    # Flujo cero a 0 y percentil 99.5 a 1
    tope = np.percentile(spectrum, 99.5)
    if tope > 0:
        spectrum /= tope

    # Ruido atenuado por el desvanecimiento, para que los extremos sigan apagados
    spectrum += rng.normal(0.0, rng.uniform(*noise_range), size=width) * curve

    # Exposicion, recorte y escala a [baseline, 255]
    spectrum *= rng.uniform(*exposure_range)
    spectrum = baseline + np.clip(spectrum, 0, 1) * (255 - baseline)

    return _intensity_function(spectrum.astype(np.uint8))


def _fading_curve(width:int, fading:Fading) -> NDArray[np.float64]:
    """Curva multiplicativa de desvanecimiento hacia los extremos del espectro.

    Parametros:
    - width {int}: largo de la curva.
    - fading {Fading}: tipo de desvanecimiento.

    Return:
    - {NDArray[np.float64]}: factor de cada columna, con maximo 1. Todo unos si fading
    es Fading.NO.
    """
    match fading:
        case Fading.GAUSSIAN:
            x_idx = np.arange(width)
            center = width * 0.5
            sigma = width * 0.3  # Ancho
            return np.exp(- (x_idx - center)**2 / (2 * sigma**2))
        case Fading.PLANCK:
            lambdas = np.linspace(0.2, 0.8, width)
            curve = planck_like(lambdas, T=0.5)
            return (curve - np.min(curve)) / (np.max(curve) - np.min(curve)) # Normalizar
        case Fading.PLANCK_BOTH_ENDS:
            curve = _fading_curve(width, Fading.PLANCK)
            pico = int(np.argmax(curve))
            # Despues del pico se reescala para que el extremo derecho tambien llegue a 0
            curve[pico:] = (curve[pico:] - curve[-1]) / (curve[pico] - curve[-1])
            return curve
    return np.ones(width)


def _intensity_function(spectrum:NDArray[np.uint8]) -> Callable[[int], int]:
    """Envuelve un espectro ya muestreado en la funcion de intensidad que pinta una parte.

    Parametros:
    - spectrum {NDArray[np.uint8]}: intensidad de cada columna, de largo width.

    Return:
    - {Callable[[int], int]}: funcion que dado un valor entero informa la intensidad
    que le corresponde.
    """
    width = len(spectrum)

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
