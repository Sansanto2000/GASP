"""Modulo de compatibilidad.

El contenido se repartio en modulos por responsabilidad (`drawing`, `spectra`, `noise`,
`labels`, `geometry`, `debug`). Este archivo reexporta todo para no romper los imports
existentes, tanto los de este repo como los de los scripts de entrenamiento externos.

Para codigo nuevo importar del modulo especifico, o directamente de `gsssp`.
Se puede eliminar cuando no queden imports de `gsssp.observationArtist` dando vueltas.
"""

from gsssp.debug import visualize_observations
from gsssp.drawing import draw_observation
from gsssp.geometry import (
    ComponentLimit,
    ObservationLimit,
    define_observation_components_limits,
    define_observations_limits,
    max_height_for_canvas,
    max_width_for_canvas,
    rotated_aabb,
)
from gsssp.labels import (
    edges_of_labels_relxywh,
    label_dict_to_yolov11_format,
    label_list_to_yolov11_format,
)
from gsssp.noise import Position, add_plate_edge, add_realistic_noise
from gsssp.spectra import Fading, planck_like, spectral_function

__all__ = [
    "ComponentLimit",
    "Fading",
    "ObservationLimit",
    "Position",
    "add_plate_edge",
    "add_realistic_noise",
    "define_observation_components_limits",
    "define_observations_limits",
    "draw_observation",
    "edges_of_labels_relxywh",
    "label_dict_to_yolov11_format",
    "label_list_to_yolov11_format",
    "max_height_for_canvas",
    "max_width_for_canvas",
    "planck_like",
    "rotated_aabb",
    "spectral_function",
    "visualize_observations",
]
