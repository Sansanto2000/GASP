"""Conversion de etiquetas a los formatos de texto que consumen los modelos."""
from enum import Enum


class LabelClass(Enum):
    """Clases que puede etiquetar el generador.

    Los indices son fijos: no se compactan segun que clases se pidan, para que el
    data.yaml de Yolo sea el mismo sin importar la configuracion de la corrida.
    Las dos lamparas de comparacion de una observacion comparten la clase LAMP.
    """
    OBSERVATION = 0
    SCIENCE = 1
    LAMP = 2


class LabelFormat(Enum):
    """Esquema de etiqueta a producir.

    - AABB: `<class> <x> <y> <w> <h>`, caja alineada a los ejes.
    - OBB: `<class> <x1> <y1> ... <x4> <y4>`, las 4 esquinas de la caja inclinada.
    """
    AABB = 0
    OBB = 1


def _clase(class_id) -> int:
    """Normaliza el indice de clase a entero.

    Las etiquetas salen del tensor del generador, que es float32, asi que la clase llega
    como 0.0 y sin esto se escribiria asi en el archivo. El formato Yolo espera un entero.
    Se redondea en vez de truncar para no perder un indice por error de representacion.

    Parametros:
    - class_id: indice de clase, entero o flotante.

    Return:
    - {int}: indice de clase como entero.
    """
    return int(round(float(class_id)))

def _yolov11_aabb(class_id, x_center, y_center, width, height) -> str:
    """Arma la linea de texto de una etiqueta AABB en formato Yolov11.

    Unico lugar donde vive el formato AABB: las variantes por tipo de entrada solo
    extraen los campos y delegan aca. Su hermana para OBB es `_yolov11_obb`.

    Parametros:
    - class_id: indice de la clase etiquetada. Se emite como entero: el tensor de salida
    del generador es float32, y una clase escrita como "0.0" no respeta el formato.
    - x_center, y_center: centro normalizado [0-1].
    - width, height: dimensiones normalizadas [0-1].

    Return:
    - {str}: informacion de la etiqueta en formato textual.
    """
    return f"{_clase(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def _yolov11_obb(class_id, corners) -> str:
    """Arma la linea de texto de una etiqueta OBB en formato Yolov11.

    Hermana de `_yolov11_aabb`: unico lugar donde vive el formato de 4 esquinas.

    Parametros:
    - class_id: indice de la clase etiquetada. Se emite como entero, igual que en AABB.
    - corners {Sequence[Number]}: 8 valores normalizados [0-1], como
    x1, y1, x2, y2, x3, y3, x4, y4. El orden de las esquinas no importa: Yolo deriva
    la caja con cv2.minAreaRect, que es independiente del orden de los puntos.

    Return:
    - {str}: informacion de la etiqueta en formato textual.
    """
    if len(corners) != 8:
        raise ValueError(f"Una etiqueta OBB necesita 8 valores, se recibieron {len(corners)}.")
    return f"{_clase(class_id)} " + " ".join(f"{c:.6f}" for c in corners)

def label_dict_to_yolov11_aabb_format(label) -> str:
    """Recibe la informacion de una etiqueta en formato dict y la convierte a un
    string en formato Yolov11.

    Parametros:
    - label {dict[str, Number]}: informacion de la etiqueta a parsear.

    Return:
    - {str}: información de la etiqueta en formato textual
    """
    return _yolov11_aabb(
        label["class_id"], label["x_center_norm"], label["y_center_norm"],
        label["width_norm"], label["height_norm"],
    )

def label_list_to_yolov11_aabb_format(label) -> str:
    """Recibe la informacion de una etiqueta en formato list y la convierte a un
    string en formato Yolov11.

    Parametros:
    - label {Sequence[Number]}: informacion de la etiqueta a parsear.

    Return:
    - {str}: información de la etiqueta en formato textual
    """
    return _yolov11_aabb(label[0], label[1], label[2], label[3], label[4])

def label_dict_to_yolov11_obb_format(label) -> str:
    """Recibe la informacion de una etiqueta en formato dict y la convierte a un
    string en formato Yolov11 OBB.

    Parametros:
    - label {dict[str, Number]}: informacion de la etiqueta a parsear. Debe traer
    la clave `corners_norm`, que produce `draw_observation`.

    Return:
    - {str}: información de la etiqueta en formato textual
    """
    return _yolov11_obb(label["class_id"], label["corners_norm"])

def label_list_to_yolov11_obb_format(label) -> str:
    """Recibe la informacion de una etiqueta en formato list y la convierte a un
    string en formato Yolov11 OBB.

    Parametros:
    - label {Sequence[Number]}: clase seguida de las 8 coordenadas normalizadas.

    Return:
    - {str}: información de la etiqueta en formato textual
    """
    return _yolov11_obb(label[0], label[1:9])

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
