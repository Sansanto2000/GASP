"""Conversion de etiquetas a los formatos de texto que consumen los modelos."""

def _yolov11(class_id, x_center, y_center, width, height) -> str:
    """Arma la linea de texto de una etiqueta AABB en formato Yolov11.

    Unico lugar donde vive el formato: las variantes por tipo de entrada solo extraen
    los campos y delegan aca. Cuando se sume OBB, el formato nuevo se agrega al lado.

    Parametros:
    - class_id: indice de la clase etiquetada.
    - x_center, y_center: centro normalizado [0-1].
    - width, height: dimensiones normalizadas [0-1].

    Return:
    - {str}: informacion de la etiqueta en formato textual.
    """
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def label_dict_to_yolov11_format(label) -> str:
    """Recibe la informacion de una etiqueta en formato dict y la convierte a un
    string en formato Yolov11.

    Parametros:
    - label {dict[str, Number]}: informacion de la etiqueta a parsear.

    Return:
    - {str}: información de la etiqueta en formato textual
    """
    return _yolov11(
        label["class_id"], label["x_center_norm"], label["y_center_norm"],
        label["width_norm"], label["height_norm"],
    )

def label_list_to_yolov11_format(label) -> str:
    """Recibe la informacion de una etiqueta en formato list y la convierte a un
    string en formato Yolov11.

    Parametros:
    - label {Sequence[Number]}: informacion de la etiqueta a parsear.

    Return:
    - {str}: información de la etiqueta en formato textual
    """
    return _yolov11(label[0], label[1], label[2], label[3], label[4])

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

# Alias de compatibilidad: los nombres viejos en camelCase siguen funcionando.
# Para codigo nuevo usar las variantes en snake_case.
labelDictToYolov11Format = label_dict_to_yolov11_format
labelListToYolov11Format = label_list_to_yolov11_format
