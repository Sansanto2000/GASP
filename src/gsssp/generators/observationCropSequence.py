"""Generador de recortes de observaciones, etiquetados por componente."""
import numpy as np
import cv2
import tensorflow as tf

from gsssp.labels import LabelClass, LabelFormat
from gsssp.generators.spectrumLabeledSequence import OutputFormat, SpectrumLabeledSequence


class ObservationCropSequence(SpectrumLabeledSequence):
  """Produce recortes de observaciones individuales, no placas enteras.

  Genera la placa igual que `SpectrumLabeledSequence` (mismo ruido, mismos bordes,
  mismas observaciones) pero devuelve el recorte de cada observacion por separado,
  con las etiquetas de sus componentes referidas al recorte.

  Existe como generador aparte y no como una bandera del otro porque alimenta a un
  modelo distinto: uno detecta observaciones sobre la placa, el otro detecta ciencia
  y lamparas dentro de una observacion ya recortada.

  El recorte lleva un margen aleatorio e independiente por lado, para que se parezca
  a uno hecho a mano: si el margen fuera simetrico, la observacion quedaria siempre
  centrada y el modelo aprenderia esa posicion en vez de reconocer el objeto.

  Solo se etiquetan los componentes de la observacion recortada. Si asoma parte de una
  observacion vecina, queda sin etiquetar a proposito: funciona como negativo dificil,
  y el modelo aprende a no confundirla con la propia. Es la misma idea que el umbral de
  area visible de `split_dota` en Ultralytics, que descarta lo que se ve poco.

  Params, ademas de los de SpectrumLabeledSequence:
  - crop_margin_range: rango porcentual del margen que se agrega a cada lado del
  recorte, respecto del tamaño de la observacion. Se sortea un valor por lado. El margen
  se acota al borde de la placa: si la observacion esta pegada a un borde, ese lado
  queda con menos margen del sorteado. Default (0.0, 0.10).
  - crop_resize_shape: dimensiones (ancho, alto) a las que se lleva cada recorte, o
  None para devolver los recortes en su tamaño original. Con None los recortes tienen
  distinto tamaño entre si, asi que el lote se devuelve como lista y no como arreglo
  apilado: sirve para volcar a disco, no para alimentar un modelo directamente.
  Default (320, 320).
  - label_classes: por defecto las dos clases de componente, que es lo que este
  generador esta pensado para etiquetar.
  """

  def __init__(
      self, *,
      crop_margin_range = (0.0, 0.10),
      crop_resize_shape = (320, 320),
      label_classes = (LabelClass.SCIENCE, LabelClass.LAMP),
      **kwargs
    ):
    super().__init__(label_classes=label_classes, **kwargs)
    self.crop_margin_range = crop_margin_range
    self.crop_resize_shape = crop_resize_shape

  def _recortar(self, img, label, rng):
    """Recorta una observacion de la placa y reubica las etiquetas de sus componentes.

    Parametros:
    - img {NDArray[np.uint8]}: la placa completa, en escala de grises.
    - label {dict}: etiqueta de la observacion, con sus `components`.
    - rng {np.random.Generator}: generador aleatorio.

    Return:
    - {NDArray[np.uint8]}: el recorte, sin redimensionar.
    - {list[dict]}: etiquetas de los componentes, normalizadas respecto del recorte.
    """
    alto, ancho = img.shape[:2]

    # Caja de la observacion en pixeles de la placa.
    x0 = (label["x_center_norm"] - label["width_norm"] / 2) * ancho
    x1 = (label["x_center_norm"] + label["width_norm"] / 2) * ancho
    y0 = (label["y_center_norm"] - label["height_norm"] / 2) * alto
    y1 = (label["y_center_norm"] + label["height_norm"] / 2) * alto
    ancho_obs, alto_obs = x1 - x0, y1 - y0

    # Un margen por lado. El sorteo es siempre sobre el rango completo, y recien
    # despues se acota a lo que hay: asi el sesgo lo pone el borde de la placa y no
    # el sorteo, que es lo que pasa con un recorte hecho a mano contra el borde.
    m_izq, m_der = rng.uniform(*self.crop_margin_range, size=2) * ancho_obs
    m_arr, m_aba = rng.uniform(*self.crop_margin_range, size=2) * alto_obs

    rx0 = int(max(0, round(x0 - m_izq)))
    rx1 = int(min(ancho, round(x1 + m_der)))
    ry0 = int(max(0, round(y0 - m_arr)))
    ry1 = int(min(alto, round(y1 + m_aba)))
    # Un recorte degenerado no sirve para nada: se garantiza al menos un pixel.
    rx1 = max(rx1, rx0 + 1)
    ry1 = max(ry1, ry0 + 1)

    recorte = img[ry0:ry1, rx0:rx1]
    rec_alto, rec_ancho = recorte.shape[:2]

    # Reubicar cada componente: de coordenadas de placa a coordenadas del recorte.
    etiquetas = []
    for comp in label["components"]:
      esquinas = np.array(comp["corners_norm"], dtype=np.float64).reshape(4, 2)
      esquinas[:, 0] = (esquinas[:, 0] * ancho - rx0) / rec_ancho
      esquinas[:, 1] = (esquinas[:, 1] * alto - ry0) / rec_alto
      xs, ys = esquinas[:, 0], esquinas[:, 1]
      etiquetas.append({
          "class_id": comp["class_id"],
          "x_center_norm": (xs.min() + xs.max()) / 2,
          "y_center_norm": (ys.min() + ys.max()) / 2,
          "width_norm": xs.max() - xs.min(),
          "height_norm": ys.max() - ys.min(),
          "corners_norm": [c for punto in esquinas for c in punto],
      })
    return recorte, etiquetas

  # Obtener el lote numero idx
  def __getitem__(self, idx):

    # Misma derivacion que el generador de placas: el lote es funcion pura del indice.
    rng = np.random.default_rng((self.seed, idx))

    batch_x = []
    batch_y = []

    # Se generan placas hasta juntar batch_size recortes, aprovechando todas las
    # observaciones de cada una. Generar una placa entera para extraer un solo recorte
    # seria varias veces mas caro.
    while len(batch_x) < self.batch_size:
      img, labels = self.generar_placa(rng)
      for label in labels:
        if len(batch_x) >= self.batch_size:
          break
        recorte, etiquetas = self._recortar(img, label, rng)
        if self.crop_resize_shape is not None:
          recorte = cv2.resize(recorte, (self.crop_resize_shape[0], self.crop_resize_shape[1]))
        batch_x.append(np.repeat(recorte[:, :, None], 3, axis=2))

        pedidas = {c.value for c in self.label_classes}
        boxes_img, classes_img = [], []
        for et in etiquetas:
          if et["class_id"] not in pedidas:
            continue
          if self.label_format is LabelFormat.OBB:
            boxes_img.append(list(et["corners_norm"]))
          else:
            boxes_img.append([
                et["x_center_norm"],
                et["y_center_norm"],
                et["width_norm"],
                et["height_norm"],
            ])
          classes_img.append(et["class_id"])
        batch_y.append({"boxes": boxes_img, "classes": classes_img})

    ### Formato de salida final ###
    # Sin redimensionar, los recortes tienen distinto tamaño y no se pueden apilar en un
    # arreglo: se devuelve la lista tal cual, que es lo que sirve para volcar a disco.
    samples = batch_x if self.crop_resize_shape is None else np.array(batch_x)
    boxes_batch = [item["boxes"] for item in batch_y]
    classes_batch = [item["classes"] for item in batch_y]
    match(self.output_format):
      case OutputFormat.LIST:
        combined = []
        for boxes, classes in zip(boxes_batch, classes_batch):
            combined.append([[cls, *box] for cls, box in zip(classes, boxes)])
        targets = tf.ragged.constant(combined, dtype=tf.float32)
      case OutputFormat.DICT:
        targets = {
            "boxes": tf.ragged.constant(boxes_batch, dtype=tf.float32),
            "classes": tf.ragged.constant(classes_batch, dtype=tf.float32),
        }
    return samples, targets
