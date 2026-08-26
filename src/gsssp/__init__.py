"""GASP - Generator for Astronomical Spectroscopic Plates.

Generador de imagenes sinteticas de escaneos de placas espectroscopicas, con etiquetas
listas para entrenar modelos de deteccion.
"""

# Dibujado de observaciones
from .drawing import drawObservation, draw_observation

# Espectros sinteticos
from .spectra import Fading, planck_like, spectral_function

# Ruido de escaneo y bordes de placa
from .noise import Position, add_plate_edge, add_realistic_noise

# Etiquetas
from .labels import (
    edges_of_labels_relxywh,
    labelDictToYolov11Format,
    labelListToYolov11Format,
    label_dict_to_yolov11_format,
    label_list_to_yolov11_format,
)

# Geometria (camino OBB)
from .geometry import (
    ComponentLimit,
    ObservationLimit,
    define_observation_components_limits,
    define_observations_limits,
    max_height_for_canvas,
    max_width_for_canvas,
    rotated_aabb,
)

# Depuracion
from .debug import visualize_observations

# Generador compatible con tensorflow
from .generators.spectrumLabeledSequence import OutputFormat, SpectrumLabeledSequence

__all__ = [
    # dibujado
    "draw_observation",
    "drawObservation",  # alias historico
    # espectros
    "Fading",
    "planck_like",
    "spectral_function",
    # ruido
    "Position",
    "add_plate_edge",
    "add_realistic_noise",
    # etiquetas
    "edges_of_labels_relxywh",
    "label_dict_to_yolov11_format",
    "label_list_to_yolov11_format",
    "labelDictToYolov11Format",  # alias historico
    "labelListToYolov11Format",  # alias historico
    # geometria
    "ComponentLimit",
    "ObservationLimit",
    "define_observation_components_limits",
    "define_observations_limits",
    "max_height_for_canvas",
    "max_width_for_canvas",
    "rotated_aabb",
    # depuracion
    "visualize_observations",
    # generador
    "OutputFormat",
    "SpectrumLabeledSequence",
]
