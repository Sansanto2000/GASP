"""GASP - Generator for Astronomical Spectroscopic Plates.

Generador de imagenes sinteticas de escaneos de placas espectroscopicas, con etiquetas
listas para entrenar modelos de deteccion.
"""

# Dibujado de observaciones
from .drawing import draw_observation

# Espectros sinteticos y a partir de LAMOST
from .spectra import Fading, lamost_spectral_function, planck_like, spectral_function
from .lamost import load_lamost_spectra, sample_lamost_window

# Ruido de escaneo y bordes de placa
from .noise import Position, add_background_field, add_plate_edge, add_realistic_noise

# Anotaciones manuscritas
from .handwriting import add_observation_annotations

# Etiquetas
from .labels import (
    LabelClass,
    LabelFormat,
    edges_of_labels_relxywh,
    label_dict_to_yolov11_aabb_format,
    label_dict_to_yolov11_obb_format,
    label_list_to_yolov11_aabb_format,
    label_list_to_yolov11_obb_format,
)

# Geometria (camino OBB)
from .geometry import (
    ObservationLimit,
    define_observations_limits,
    max_height_for_canvas,
    max_width_for_canvas,
    rotated_aabb,
)

# Depuracion
from .debug import visualize_observations

# Generador compatible con tensorflow
from .generators.observationCropSequence import ObservationCropSequence
from .generators.spectrumLabeledSequence import OutputFormat, SpectrumLabeledSequence

__all__ = [
    # dibujado
    "draw_observation",
    # espectros
    "Fading",
    "lamost_spectral_function",
    "load_lamost_spectra",
    "planck_like",
    "sample_lamost_window",
    "spectral_function",
    # ruido
    "Position",
    "add_background_field",
    "add_plate_edge",
    "add_realistic_noise",
    # texto manuscrito
    "add_observation_annotations",
    # etiquetas
    "LabelClass",
    "LabelFormat",
    "edges_of_labels_relxywh",
    "label_dict_to_yolov11_aabb_format",
    "label_dict_to_yolov11_obb_format",
    "label_list_to_yolov11_aabb_format",
    "label_list_to_yolov11_obb_format",
    # geometria
    "ObservationLimit",
    "define_observations_limits",
    "max_height_for_canvas",
    "max_width_for_canvas",
    "rotated_aabb",
    # depuracion
    "visualize_observations",
    # generadores
    "ObservationCropSequence",
    "OutputFormat",
    "SpectrumLabeledSequence",
]
