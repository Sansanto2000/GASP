"""Conversion de etiquetas a los formatos de texto que consumen los modelos."""

def labelDictToYolov11Format(label) -> str:
    """Recibe la informacion de una etiqueta en formato dict y la convierte a un 
    string en formato Yolov11.

    Parametros:
    - label {dict[str, Number]}: informacion de la etiqueta a parsear.

    Return:
    - {str}: información de la etiqueta en formato textual
    """
    return f"{label['class_id']} {label['x_center_norm']:.6f} {label['y_center_norm']:.6f} {label['width_norm']:.6f} {label['height_norm']:.6f}"

def labelListToYolov11Format(label) -> str:
    """Recibe la informacion de una etiqueta en formato list y la convierte a un 
    string en formato Yolov11.

    Parametros:
    - label {dict[str, Number]}: informacion de la etiqueta a parsear.

    Return:
    - {str}: información de la etiqueta en formato textual
    """
    return f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}"

def edges_of_labels_relxywh(labels, alto, ancho):
        """dado un conjunto de etiquetas en formato dict relxywh y un
        ancho y alto de imagen determina en pixeles los limites 
        [x_min, x_max, y_min, y_max] donde se mueven las etiquetas.

        Args:
            labels ([type]): conjunto de etiquetas en formato dict relxywh.
            alto ([type]): alto de la imagen en pixeles.
            ancho ([type]): ancho de la imagen en pixeles.

        Returns:
            int[]: [x_min, x_max, y_min, y_max] limites en pixeles donde 
            se mueven las etiquetas.
        """
        min_x = ancho
        max_x = 0
        min_y = alto
        max_y = 0
        for label in labels:
          x_center = label['x_center_norm'] * ancho
          y_center = label['y_center_norm'] * alto
          width = label['width_norm'] * ancho
          height = label['height_norm'] * alto
          
          x_min = x_center - width/2
          x_max = x_center + width/2
          y_min = y_center - height/2
          y_max = y_center + height/2
          if x_min < min_x:
            min_x = x_min
          if x_max > max_x:
            max_x = x_max
          if y_max > max_y:
            max_y = y_max
          if y_min < min_y:
            min_y = y_min
        return [min_x, max_x, min_y, max_y]
