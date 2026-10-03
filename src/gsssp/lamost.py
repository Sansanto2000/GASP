"""Espectros estelares reales de LAMOST DR12 LRS, para construir el espectro de ciencia.

Los datos viven en Hugging Face (`Sansanto/gasp-lamost-dr12-lrs`) y se descargan la
primera vez que se piden, igual que EMNIST en `handwriting.py`: no hacen falta para
importar el paquete, solo para generar una placa.
"""
import numpy as np
from numpy.typing import NDArray

# Version fijada por tag y verificada por sha256: la misma semilla tiene que dar el mismo
# espectro, asi que no se puede seguir a `main` del dataset, que puede cambiar.
_REPO = "https://huggingface.co/datasets/Sansanto/gasp-lamost-dr12-lrs"
_REVISION = "100k-1"
_ARCHIVO = "lrs_dr12_intensidades_uint8_delta.npz"
_SHA256 = "7ef37517d2a03a8237b1d2c23f50cceb9b4f78be3f4d289ecc6272a611ba480c"

# Mas al azul de 3900 A domina el ruido en LRS, asi que los tramos arrancan desde aca.
LAMBDA_MIN = 3900.0

_ESPECTROS = None  # (intensidades, desde, hasta, coeff0, coeff1)

def load_lamost_spectra():
    """Descarga (la primera vez), decodifica y cachea a nivel de modulo los espectros.

    El archivo queda en ~/.keras/datasets/gasp_lamost (240 MB) y se verifica por sha256.
    En memoria se decodifica en el lugar, sin copia: ~390 MB, una sola vez por proceso.
    Son datos inmutables, no estado que afecte la reproducibilidad: que espectro y que
    tramo usar se sortea despues con el rng de la placa.

    Para un entorno sin internet, correr esto una vez de antemano con conexion.

    Return:
    - {NDArray[np.uint8]}: intensidades con forma (N, 3920). 255 equivale a 1,5 veces
    el percentil 99 del flujo de cada espectro.
    - {NDArray[int]}: primera columna utilizable de cada espectro (>= LAMBDA_MIN).
    - {NDArray[int]}: columna siguiente a la ultima con datos de cada espectro.
    - {float}, {float}: coeficientes de la grilla, lambda_i = 10**(coeff0 + coeff1*i) A.
    """
    global _ESPECTROS
    if _ESPECTROS is None:
        from keras.utils import get_file
        ruta = get_file(
            fname=_ARCHIVO,
            origin=f"{_REPO}/resolve/{_REVISION}/{_ARCHIVO}",
            file_hash=_SHA256,
            hash_algorithm="sha256",
            cache_subdir="datasets/gasp_lamost",
        )
        with np.load(ruta) as d:
            intensidades = d["intensidades_delta"]
            # Vienen codificadas por diferencias entre pixeles vecinos (sin perdida).
            np.cumsum(intensidades, axis=1, dtype=np.uint8, out=intensidades)
            coeff0, coeff1 = float(d["coeff0"]), float(d["coeff1"])
            columna_min = int(np.ceil((np.log10(LAMBDA_MIN) - coeff0) / coeff1))
            desde = np.maximum(d["inicio"].astype(np.int64), columna_min)
            hasta = d["inicio"].astype(np.int64) + d["npix"]
        _ESPECTROS = (intensidades, desde, hasta, coeff0, coeff1)
    return _ESPECTROS

def sample_lamost_window(
    window_range=(3000, 6000),
    *, rng: np.random.Generator = None
) -> NDArray[np.uint8]:
    """Sortea un espectro del dataset y un tramo contiguo de el.

    Parametros:
    - window_range {tuple[float, float]}?: rango del ancho del tramo, en angstrom. Se
    recorta a lo que cubre cada espectro desde LAMBDA_MIN (entre ~4870 y ~5100 A en todo
    el dataset), asi que el maximo por defecto equivale a "hasta el espectro completo".
    Default (3000, 6000).
    - rng {np.random.Generator}?: generador aleatorio a usar. Si no se pasa se crea uno
    sin semilla. Recibirlo permite que el resultado sea reproducible y seguro entre hilos.

    Return:
    - {NDArray[np.uint8]}: intensidades del tramo, de azul a rojo.
    """
    if rng is None:
        rng = np.random.default_rng()

    intensidades, desde, hasta, coeff0, coeff1 = load_lamost_spectra()
    i = int(rng.integers(0, len(intensidades)))
    # La grilla es logaritmica: un ancho fijo en angstrom no es un largo fijo en pixeles,
    # asi que se sortea en longitud de onda y recien despues se pasa a columnas.
    lam_min = 10 ** (coeff0 + coeff1 * desde[i])
    lam_max = 10 ** (coeff0 + coeff1 * (hasta[i] - 1))
    disponible = lam_max - lam_min
    ancho = rng.uniform(min(window_range[0], disponible), min(window_range[1], disponible))
    lam_inicio = rng.uniform(lam_min, lam_max - ancho)
    columna = lambda lam: (np.log10(lam) - coeff0) / coeff1
    inicio = max(int(np.floor(columna(lam_inicio))), int(desde[i]))
    fin = min(int(np.ceil(columna(lam_inicio + ancho))) + 1, int(hasta[i]))
    return intensidades[i, inicio:fin]
