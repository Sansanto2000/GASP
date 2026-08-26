from keras.utils import Sequence
import random
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
from gsssp.labels import edges_of_labels_relxywh
from gsssp.noise import Position, add_plate_edge, add_realistic_noise

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
  - band_intensity_range: rango de ruido de banda horizontal.
  - speck_count_range: rango de cantidad de manchas de polvo.
  - speck_size_range: rango de tamaño de las manchas de polvo.
  - blur_kernel_size_options: lista de opciones enteras para el tamaño del kernel de 
  desenfoque.
  - prob_edge: probabilidad de que se añada un borde a la placa.
  - batch_size: cantidad de elementos por lote.
  - resize_shape: dimensiones (ancho, alto) para las imagenes finales.
  - output_format: formato de datos de salida.
  - batchs_per_sequence: cantidad de lotes a producir en una secuencia.
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
      batchs_per_sequence = 100,
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
    self.band_intensity_range = band_intensity_range
    self.speck_count_range = speck_count_range
    self.speck_size_range = speck_size_range
    self.blur_kernel_size_options = blur_kernel_size_options
    self.batch_size = batch_size
    self.resize_shape = resize_shape
    self.violin_line_include = violin_line_include
    self.output_format = output_format
    self.batchs_per_sequence = batchs_per_sequence
    self.violin_intensity_range = violin_intensity_range
    self.violin_length_range = violin_length_range
    self.prob_edge = prob_edge
    self.rng = np.random.default_rng()

  # Number of batch in the Sequence.
  def __len__(self):
    #return math.ceil((self.max_index - (self.min_index + self.lookback)) / self.batch_size)
    return self.batchs_per_sequence

  # Obtener el lote numero idx
  def __getitem__(self, idx):

    batch_x = []
    batch_y = []
    
    for i in range(self.batch_size):
      ### Canvas ###
      # Dimensiones.
      alto = random.randint(*self.height_range)
      ancho = random.randint(*self.width_range)
      # Imagen base oscura completa.
      gray_value = np.random.randint(self.gray_value_range[0]*255, self.gray_value_range[1]*255)
      img = np.full((alto, ancho, 3), gray_value, dtype=np.uint8)
      
      ### Definir limites de las observaciones ###
      observations_limits = define_observations_limits(alto, ancho, self.rng)

      ### Observacion ###
      # Ancho de la observacion que varia en relacion al ancho total disponible.
      obs_width = random.randint(int(ancho*0.4), int(ancho*0.95))
      # Alto total de la observacion que varia en relacion al ancho de la misma.
      obs_heigth = random.randint(int(alto*0.1), int(alto*0.8))
      # Inclinacion de la observacion.
      angle = random.randint(*self.angle_range)
      # Recortar ancho y alto para que la caja envolvente de la observacion inclinada entre
      # en el canvas. Si no, la etiqueta normalizada supera 1 y Yolo descarta la imagen entera.
      obs_width = max(1, min(obs_width, int(max_width_for_canvas(angle, alto, ancho))))
      obs_heigth = max(1, min(obs_heigth, int(max_height_for_canvas(obs_width, angle, alto, ancho))))
      # Que tan anchas van a ser los espectros de lampara en relacion al espectro de ciencia
      openingLamp = random.uniform(*self.opening_lamp_range)
      # Cuanto espacio vacio hay entre cada lampara y el espectro de ciencia.
      distanceBetweenParts = random.uniform(*self.distance_between_components_range)

      ### Grupo de observaciones ###
      # Distancia entre distintas observaciones
      distanceBetweenObservations = random.uniform(
        obs_heigth*self.distance_between_observations_range[0], 
        obs_heigth*self.distance_between_observations_range[1]
      )
      # Cantidad de observaciones que entran en la imagen
      max_observations = math.floor(alto*0.95/(obs_heigth+distanceBetweenObservations/2))
      # Cuantas observaciones se dibujaran en una la imagen
      n_observations = min(max_observations, random.randint(1, self.cant_observations_max))

      ### Definir posiciones ###
      # Posiciones donde ser realizara el dibujo centradas en alto
      noise_horizontal = self.noise_horizontal # Irregularidad porcentual horizontal maxima
      noise_vertical = self.noise_vertical # Irregularidad porcentual veartical maxima
      unit = obs_heigth + distanceBetweenObservations# Espacio a considerar por observación
      targets = []
      for i in range(n_observations):
        pos_y = (alto/2) - (n_observations/2)*unit + unit/2 + i*unit
        coor = {
          "x": ancho/2 + random.uniform(-noise_horizontal, noise_horizontal), 
          "y": pos_y + random.uniform(-noise_vertical, noise_vertical), 
        }
        targets.append(coor)
      # eliminar targets con probabilidad 0.10
      filtered_targets = [t for t in targets if random.random() > 0.10]
      # garantizar que quede al menos 1
      if len(filtered_targets) == 0 and len(targets) > 0:
        filtered_targets.append(random.choice(targets))
      targets = filtered_targets

      ### Dibujar ###
      labels = []
      for coor in targets:
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
        )
        labels.append(label)

      ### Bordes de la placa ###
      if(self.prob_edge > 0 and random.random() < self.prob_edge):
        [x_min, x_max, y_min, y_max ] = edges_of_labels_relxywh(labels, alto, ancho)
        side = random.choice([Position.RIGHT, Position.LEFT, Position.TOP, Position.BOTTOM])
        img = add_plate_edge(img, (x_min, x_max, y_min, y_max), side)

      ### Ruido y manchas ###
      # Ruido gaussiano general para la imagen de la placa
      gaussian_std = random.uniform(*self.gaussian_std_range)
      # Ruido de banda horizontal
      band_intensity = random.uniform(*self.band_intensity_range)
      # Cantidad de manchas de polvo
      speck_count = random.randint(*self.speck_count_range)
      # Radio maximo de las manchas de polvo
      speck_size = random.randint(*self.speck_size_range)
      # Nivel del desenfoque gaussiano
      blur_kernel_size = random.choice(self.blur_kernel_size_options)
      # Cantidad de manchas alargadas tipo "violín"
      violin_line_count = np.random.choice(
          [0, 1, 2, 3],
          p=[0.9, 0.09, 0.009, 0.001]
        ) if self.violin_line_include else 0
      # Intensidad de las manchas alargadas tipo "violín"
      violin_intensity = random.uniform(*self.violin_intensity_range)
      # Añadir ruido en la imagen
      img = add_realistic_noise(
        img,
        gaussian_std=gaussian_std,
        band_intensity=band_intensity,
        speck_count=speck_count,
        speck_size=speck_size,
        blur_ksize=blur_kernel_size,
        violin_line_count=violin_line_count,
        violin_intensity=violin_intensity,
        violin_length_range=self.violin_length_range
      )

      # Redimensionar imagen
      img = cv2.resize(img, (self.resize_shape[0], self.resize_shape[1]))
      batch_x.append(img)
      
      # Ajustar formato
      boxes_img = []
      classes_img = []
      for label in labels:
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