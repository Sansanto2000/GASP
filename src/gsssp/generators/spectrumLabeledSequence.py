from keras.utils import Sequence
import math
from enum import Enum

import numpy as np
import cv2
import tensorflow as tf

from gsssp.drawing import draw_observation
from gsssp.geometry import (
    define_observations_limits,
    max_height_for_canvas,
    max_width_for_canvas,
)
from gsssp.labels import LabelClass, LabelFormat, edges_of_labels_relxywh
from gsssp.noise import Position, add_background_field, add_plate_edge, add_realistic_noise

class OutputFormat(Enum):
  LIST = 0
  DICT = 1

"""Generador compatible con tensorflow para alimentar modelos 
de aprendizaje automatico. 
"""
class SpectrumLabeledSequence(Sequence):

  """Inicializador.
  
  Params:
  - height_range: rango de configuracion aleatoria para la altura del canvas
  - width_range: rango de configuracion aleatoria para el ancho del canvas
  - gray_value_range: rango porcentual de nivel de gris del fondo. 0 es negro, 1 es blanco.
  - angle_range: rango en grados de angulo de inclinacion de las observaciones.
  - opening_lamp_range: rango porcentual de que tan anchos seran los espectros de lampara en 
  relacion a los especros de ciencia.
  - distance_between_components_range: rango porcentual del espacio vacio entre cada 
  componente de la observacion.
  - distance_between_observations_range: rango porcentual del espacio vacio entre cada 
  observacion.
  - cant_observations_max: numero de observaciones maximo.
  - noise_horizontal: porcentaje de desviacion horizontal para el centro de una observacion.
  - noise_vertical: porcentaje de desviacion horizontal para el centro de una observacion.
  - gaussian_std_range: rango de ruido gaussiano general par la imagen generada.
  - grain_std_range: rango de intensidad del grano de emulsion, ruido de baja escala
  espacialmente correlacionado (a diferencia del ruido gaussiano de gaussian_std_range,
  que es independiente por pixel).
  - field_amplitude_range: rango de amplitud del campo de iluminacion de baja
  frecuencia que se aplica al canvas antes de dibujar las observaciones. Simula
  iluminacion despareja o vineteado del escaner. 0 no aplica ningun efecto.
  - band_intensity_range: rango de ruido de banda horizontal.
  - speck_count_range: rango de cantidad de manchas de polvo.
  - speck_size_range: rango de tamaño de las manchas de polvo.
  - blur_kernel_size_options: lista de opciones enteras para el tamaño del kernel de 
  desenfoque.
  - prob_edge: probabilidad de que se añada un borde a la placa.
  - batch_size: cantidad de elementos por lote.
  - resize_shape: dimensiones (ancho, alto) para las imagenes finales.
  - output_format: formato de datos de salida.
  - label_classes: clases a etiquetar, como tupla de LabelClass. Por defecto solo
  LabelClass.OBSERVATION. Sumando SCIENCE y LAMP se etiquetan tambien los componentes
  de cada observacion: el espectro de ciencia y las dos lamparas de comparacion.
  Los indices de clase son fijos (0, 1, 2) y no se compactan segun la seleccion.
  - label_format: esquema de etiqueta. LabelFormat.AABB produce 4 valores por caja
  (centro y tamaño); LabelFormat.OBB produce 8 (las 4 esquinas de la caja inclinada).
  Default AABB.
  - batchs_per_sequence: cantidad de lotes a producir en una secuencia.
  - seed: semilla base del generador. El lote se deriva de (seed, idx), asi que con la
  misma semilla la secuencia completa es reproducible. Default 0.
  """
  def __init__(
      self, *, 
      height_range = (1000, 4000), 
      width_range = (1000, 4000), 
      gray_value_range = (0, 0.6), 
      angle_range = (-3, 3), 
      opening_lamp_range = (0.1, 0.45), 
      distance_between_components_range = (0.001, 0.1),
      distance_between_observations_range = (0.05, 0.4), 
      cant_observations_max = 5,
      noise_horizontal = 0.01, 
      noise_vertical = 0.01, 
      gaussian_std_range= (4.0, 16.0),
      grain_std_range = (0.0, 15.0),
      field_amplitude_range = (0.0, 0.12),
      band_intensity_range = (0.0, 1.0),
      speck_count_range = (0, 10), 
      speck_size_range = (1,5), 
      blur_kernel_size_options = [1, 3, 5, 7, 9, 11, 13, 15],
      batch_size = 128,
      resize_shape = (640, 640),
      violin_line_include:bool = True,
      violin_intensity_range = (0.1, 1.0),
      violin_length_range = (0.05, 0.7),
      prob_edge = 0.1,
      output_format:OutputFormat = OutputFormat.LIST,
      label_format:LabelFormat = LabelFormat.AABB,
      label_classes = (LabelClass.OBSERVATION,),
      batchs_per_sequence = 100,
      seed = 0,
      **kwargs
    ):
    super().__init__(**kwargs)
    
    self.height_range = height_range
    self.width_range = width_range
    self.gray_value_range = gray_value_range
    self.angle_range = angle_range
    self.opening_lamp_range = opening_lamp_range
    self.distance_between_components_range =distance_between_components_range
    self.distance_between_observations_range = distance_between_observations_range
    self.cant_observations_max = cant_observations_max
    self.noise_horizontal = noise_horizontal
    self.noise_vertical = noise_vertical
    self.gaussian_std_range = gaussian_std_range
    self.grain_std_range = grain_std_range
    self.field_amplitude_range = field_amplitude_range
    self.band_intensity_range = band_intensity_range
    self.speck_count_range = speck_count_range
    self.speck_size_range = speck_size_range
    self.blur_kernel_size_options = blur_kernel_size_options
    self.batch_size = batch_size
    self.resize_shape = resize_shape
    self.violin_line_include = violin_line_include
    self.output_format = output_format
    self.label_format = label_format
    self.label_classes = tuple(label_classes)
    self.batchs_per_sequence = batchs_per_sequence
    self.violin_intensity_range = violin_intensity_range
    self.violin_length_range = violin_length_range
    self.prob_edge = prob_edge
    self.seed = seed

  # Number of batch in the Sequence.
  def __len__(self):
    #return math.ceil((self.max_index - (self.min_index + self.lookback)) / self.batch_size)
    return self.batchs_per_sequence

  def generar_placa(self, rng):
    """Genera una placa completa a resolucion original, con sus observaciones.

    Es el nucleo compartido de los generadores: `__getitem__` la redimensiona y arma el
    lote, y un generador de recortes puede tomar cada observacion a partir de lo mismo.

    Parametros:
    - rng {np.random.Generator}: generador aleatorio del que cuelga todo el sorteo.

    Return:
    - {NDArray[np.uint8]}: la placa en escala de grises, sin redimensionar.
    - {list[dict]}: una etiqueta por observacion, cada una con sus componentes.
    """
    ### Canvas ###
    # Dimensiones.
    alto = int(rng.integers(self.height_range[0], self.height_range[1] + 1))
    ancho = int(rng.integers(self.width_range[0], self.width_range[1] + 1))
    # Imagen base oscura completa. Se trabaja en escala de grises: la placa es monocroma,
    # asi que arrastrar tres canales identicos durante el pipeline triplica memoria y
    # computo sin agregar informacion. La expansion a tres canales se hace al final.
    gray_value = int(rng.integers(self.gray_value_range[0]*255, self.gray_value_range[1]*255))
    img = np.full((alto, ancho), gray_value, dtype=np.uint8)
    # Campo de iluminacion de baja frecuencia, multiplicativo. Se aplica antes de dibujar
    # las observaciones para que el gradiente tambien las afecte a ellas.
    field_amplitude = rng.uniform(*self.field_amplitude_range)
    img = add_background_field(img, field_amplitude, rng=rng)

    ### Definir limites de las observaciones ###
    observations_limits = define_observations_limits(alto, ancho, rng)

    ### Observacion ###
    # Ancho de la observacion que varia en relacion al ancho total disponible.
    obs_width = int(rng.integers(int(ancho*0.4), int(ancho*0.95) + 1))
    # Alto total de la observacion que varia en relacion al ancho de la misma.
    obs_heigth = int(rng.integers(int(alto*0.1), int(alto*0.8) + 1))
    # Inclinacion de la observacion.
    angle = int(rng.integers(self.angle_range[0], self.angle_range[1] + 1))
    # Recortar ancho y alto para que la caja envolvente de la observacion inclinada entre
    # en el canvas. Si no, la etiqueta normalizada supera 1 y Yolo descarta la imagen entera.
    obs_width = max(1, min(obs_width, int(max_width_for_canvas(angle, alto, ancho))))
    obs_heigth = max(1, min(obs_heigth, int(max_height_for_canvas(obs_width, angle, alto, ancho))))
    # Que tan anchas van a ser los espectros de lampara en relacion al espectro de ciencia
    openingLamp = rng.uniform(*self.opening_lamp_range)
    # Cuanto espacio vacio hay entre cada lampara y el espectro de ciencia.
    distanceBetweenParts = rng.uniform(*self.distance_between_components_range)

    ### Grupo de observaciones ###
    # Distancia entre distintas observaciones
    distanceBetweenObservations = rng.uniform(
      obs_heigth*self.distance_between_observations_range[0], 
      obs_heigth*self.distance_between_observations_range[1]
    )
    # Cantidad de observaciones que entran en la imagen
    max_observations = math.floor(alto*0.95/(obs_heigth+distanceBetweenObservations/2))
    # Cuantas observaciones se dibujaran en una la imagen
    n_observations = min(max_observations, int(rng.integers(1, self.cant_observations_max + 1)))

    ### Definir posiciones ###
    # Posiciones donde ser realizara el dibujo centradas en alto
    noise_horizontal = self.noise_horizontal # Irregularidad porcentual horizontal maxima
    noise_vertical = self.noise_vertical # Irregularidad porcentual veartical maxima
    unit = obs_heigth + distanceBetweenObservations# Espacio a considerar por observación
    posiciones = []
    for i in range(n_observations):
      pos_y = (alto/2) - (n_observations/2)*unit + unit/2 + i*unit
      coor = {
        "x": ancho/2 + rng.uniform(-noise_horizontal, noise_horizontal),
        "y": pos_y + rng.uniform(-noise_vertical, noise_vertical),
      }
      posiciones.append(coor)
    # eliminar posiciones con probabilidad 0.10
    posiciones_filtradas = [t for t in posiciones if rng.random() > 0.10]
    # garantizar que quede al menos 1
    if len(posiciones_filtradas) == 0 and len(posiciones) > 0:
      posiciones_filtradas.append(posiciones[int(rng.integers(0, len(posiciones)))])
    posiciones = posiciones_filtradas

    ### Dibujar ###
    labels = []
    for coor in posiciones:
      img, _obs, _mask, label = draw_observation(
        img=img,
        x=coor["x"], 
        y=coor["y"],
        width=int(obs_width),
        height=int(obs_heigth),
        opening=openingLamp,
        distanceBetweenParts=distanceBetweenParts,
        angle=angle,
        baseGrey=gray_value,
        inplace=True,
        debug=False,
        rng=rng,
      )
      labels.append(label)

    ### Bordes de la placa ###
    if(self.prob_edge > 0 and rng.random() < self.prob_edge):
      [x_min, x_max, y_min, y_max ] = edges_of_labels_relxywh(labels, alto, ancho)
      side = [Position.RIGHT, Position.LEFT, Position.TOP, Position.BOTTOM][int(rng.integers(0, 4))]
      img = add_plate_edge(img, (x_min, x_max, y_min, y_max), side, rng=rng)

    ### Ruido y manchas ###
    # Ruido gaussiano general para la imagen de la placa
    gaussian_std = rng.uniform(*self.gaussian_std_range)
    # Intensidad del grano de emulsion (ruido de baja escala correlacionado)
    grain_std = rng.uniform(*self.grain_std_range)
    # Ruido de banda horizontal
    band_intensity = rng.uniform(*self.band_intensity_range)
    # Cantidad de manchas de polvo
    speck_count = int(rng.integers(self.speck_count_range[0], self.speck_count_range[1] + 1))
    # Radio maximo de las manchas de polvo
    speck_size = int(rng.integers(self.speck_size_range[0], self.speck_size_range[1] + 1))
    # Nivel del desenfoque gaussiano
    blur_kernel_size = self.blur_kernel_size_options[int(rng.integers(0, len(self.blur_kernel_size_options)))]
    # Cantidad de manchas alargadas tipo "violín"
    violin_line_count = rng.choice(
        [0, 1, 2, 3],
        p=[0.9, 0.09, 0.009, 0.001]
      ) if self.violin_line_include else 0
    # Intensidad de las manchas alargadas tipo "violín"
    violin_intensity = rng.uniform(*self.violin_intensity_range)
    # Añadir ruido en la imagen
    img = add_realistic_noise(
      img,
      gaussian_std=gaussian_std,
      grain_std=grain_std,
      band_intensity=band_intensity,
      speck_count=speck_count,
      speck_size=speck_size,
      blur_ksize=blur_kernel_size,
      violin_line_count=violin_line_count,
      violin_intensity=violin_intensity,
      violin_length_range=self.violin_length_range,
      rng=rng,
    )

    return img, labels

  # Obtener el lote numero idx
  def __getitem__(self, idx):

    # El generador aleatorio se deriva de (semilla, idx), asi que el lote es funcion pura
    # del indice: el mismo idx siempre da el mismo lote, e indices distintos dan lotes
    # distintos. Al ser un objeto propio de esta llamada, no hay estado compartido entre
    # hilos ni entre procesos.
    rng = np.random.default_rng((self.seed, idx))

    batch_x = []
    batch_y = []

    for i in range(self.batch_size):
      img, labels = self.generar_placa(rng)

      # Redimensionar imagen y recien ahi expandir a tres canales, que es lo que esperan
      # los modelos de deteccion con backbone preentrenado.
      img = cv2.resize(img, (self.resize_shape[0], self.resize_shape[1]))
      batch_x.append(np.repeat(img[:, :, None], 3, axis=2))
      
      # Expandir cada observacion a las etiquetas pedidas: la observacion entera y/o
      # sus componentes. draw_observation ya devuelve ambas cosas.
      pedidas = {c.value for c in self.label_classes}
      etiquetas = []
      for label in labels:
          if label['class_id'] in pedidas:
              etiquetas.append(label)
          etiquetas.extend(c for c in label['components'] if c['class_id'] in pedidas)

      # Ajustar formato. Segun label_format cada caja son 4 valores (centro y tamaño)
      # u 8 (las esquinas de la caja inclinada), que es lo que espera Yolo para OBB.
      boxes_img = []
      classes_img = []
      for label in etiquetas:
          if self.label_format is LabelFormat.OBB:
              boxes_img.append(list(label['corners_norm']))
          else:
              boxes_img.append([
                  label['x_center_norm'],
                  label['y_center_norm'],
                  label['width_norm'],
                  label['height_norm']
              ])
          classes_img.append(label['class_id'])
      batch_y.append({
          "boxes": boxes_img,
          "classes": classes_img
      })

    ### Fomato de salida final ###
    # Imagenes
    samples = np.array(batch_x)
    # Etiquetas
    boxes_batch = [item["boxes"] for item in batch_y]
    classes_batch = [item["classes"] for item in batch_y]
    match(self.output_format):
      case OutputFormat.LIST:
        combined = []
        for boxes, classes in zip(boxes_batch, classes_batch):
            img_labels = []
            for cls, box in zip(classes, boxes):
                img_labels.append([cls, *box])
            combined.append(img_labels)
        targets = tf.ragged.constant(combined, dtype=tf.float32)
      case OutputFormat.DICT:
        targets = {
            "boxes": tf.ragged.constant(
              boxes_batch, 
              ragged_rank=1, 
              inner_shape=(4,), 
              dtype=tf.float32
            ),
            "classes": tf.ragged.constant(classes_batch, dtype=tf.float32),
        }

    return samples, targets