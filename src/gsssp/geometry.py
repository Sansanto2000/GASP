"""Geometria de observaciones: rotacion de puntos y limites (camino OBB)."""
import math
from numbers import Number
from typing import Tuple

import numpy as np

def rotated_aabb(width:Number, height:Number, angle_degrees:Number) -> Tuple[Number,Number]:
    """Dimensiones de la caja alineada a los ejes que envuelve un rectangulo rotado.

    Parametros:
    - width {Number}: ancho del rectangulo sin rotar.
    - height {Number}: alto del rectangulo sin rotar.
    - angle_degrees {Number}: angulo de inclinacion en grados.

    Return:
    - {Tuple[Number,Number]}: (ancho, alto) de la caja envolvente.
    """
    c = abs(math.cos(math.radians(angle_degrees)))
    s = abs(math.sin(math.radians(angle_degrees)))
    return width*c + height*s, width*s + height*c

def max_width_for_canvas(angle_degrees:Number, alto:int, ancho:int) -> float:
    """Ancho maximo de una observacion para que quepa inclinada, aun con alto cero.

    Con angulos grandes una observacion muy ancha no entra en el canvas por mas fina
    que sea: su propia diagonal ya se pasa. Acotar el alto no alcanza en ese caso.

    Parametros:
    - angle_degrees {Number}: angulo de inclinacion en grados.
    - alto {int}: alto del canvas en pixeles.
    - ancho {int}: ancho del canvas en pixeles.

    Return:
    - {float}: ancho maximo admisible.
    """
    c = abs(math.cos(math.radians(angle_degrees)))
    s = abs(math.sin(math.radians(angle_degrees)))
    limite_ancho = ancho / c if c > 0 else float("inf")
    limite_alto = alto / s if s > 0 else float("inf")
    return min(limite_ancho, limite_alto)

def max_height_for_canvas(width:Number, angle_degrees:Number, alto:int, ancho:int) -> float:
    """Alto maximo de una observacion para que su caja envolvente entre en el canvas.

    Una observacion inclinada ocupa una caja alineada a los ejes mas grande que ella:
    a 18 grados el alto de la envolvente llega a ser 2.8x el alto nominal. Si esa caja
    supera el canvas, la etiqueta normalizada da width_norm o height_norm mayor a 1 y
    Yolo descarta la imagen entera al validar el dataset.

    Parametros:
    - width {Number}: ancho de la observacion en pixeles.
    - angle_degrees {Number}: angulo de inclinacion en grados.
    - alto {int}: alto del canvas en pixeles.
    - ancho {int}: ancho del canvas en pixeles.

    Return:
    - {float}: alto maximo admisible. Puede ser 0 si el ancho por si solo ya no entra.
    """
    c = abs(math.cos(math.radians(angle_degrees)))
    s = abs(math.sin(math.radians(angle_degrees)))
    # De width*s + height*c <= alto  y  width*c + height*s <= ancho
    limite_alto = (alto - width*s) / c if c > 0 else float("inf")
    limite_ancho = (ancho - width*c) / s if s > 0 else float("inf")
    return max(0.0, min(limite_alto, limite_ancho))

def rotate_point(x:Number, y, cx, cy, angle_degrees) -> Tuple[Number,Number]:
    """Rotar un punto en relacion centro segun la formula de rotacion 2D.

    Parametros:
    - x {Number}: X del punto a rotar.
    - y {Number}: Y del punto a rotar.
    - cx {Number}: X del centro.
    - cy {Number}: Y del centro.
    - angle_degrees {Number}: angulo de rotación (en grados).

    Return:
    - {Tuple[Number,Number]}: punto luego de rotar.
    """
    theta = np.radians(angle_degrees)

    # Trasladar el punto para que el centro sea el origen
    tx = x - cx
    ty = y - cy

    # Aplicar rotación
    rx = tx * np.cos(theta) - ty * np.sin(theta)
    ry = tx * np.sin(theta) + ty * np.cos(theta)

    # Trasladar de vuelta
    return rx + cx, ry + cy


class ComponentLimit:
    """Clase que representa los limites de una componente de la observacion (lampara 1, lampara 2 o espectro de ciencia).
    """
    def __init__(self, type: str, angle: int, alto: int, ancho: int, x_center: int, y_center: int, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, x4: int, y4: int):
        self.type = type
        self.angle = angle
        self.alto = alto
        self.ancho = ancho
        self.x_center = x_center
        self.y_center = y_center
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4

class ObservationLimit:
    """Clase que representa los limites de una observacion dentro de una imagen.
    """
    def __init__(self, angle: int, alto: int, ancho: int, x_center: int, y_center: int, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, x4: int, y4: int):
        self.angle = angle
        self.alto = alto
        self.ancho = ancho
        self.x_center = x_center
        self.y_center = y_center
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4
        self.science = None
        self.lamps = None
    
    def define_components_limits(self):
        """Define los limites de cada componente de la observacion (lampara 1, lampara 2 y espectro de ciencia) 
        en base a los limites generales de la observacion.
        """
        self.science = None
        self.lamps
        pass
        
    def __str__(self):
        return (
            f"ObservationLimit("
            f"angle={self.angle}, "
            f"alto={self.alto}, "
            f"ancho={self.ancho}, "
            f"x_center={self.x_center}, "
            f"y_center={self.y_center}, "
            f"x1={self.x1}, "
            f"y1={self.y1}, "
            f"x2={self.x2}, "
            f"y2={self.y2}, "
            f"x3={self.x3}, "
            f"y3={self.y3}, "
            f"x4={self.x4}, "
            f"y4={self.y4}, "
            f"science={len(self.science) if self.science else None}, "
            f"lamps={len(self.lamps) if self.lamps else None}"
        )
        
    __repr__ = __str__

def define_observation_components_limits(observation):
    pass
    
def define_observations_limits(alto, ancho, rng, min_heigth=0.03, min_width=0.1):
    """Define una lista de detalles de observaciones que entran en una imagen en base al alto y ancho de un canvas.

    Args:
        alto (int): alto de la imagen.
        ancho (int): ancho de la imagen.
        rng (np.random.Generator): generador de numeros aleatorios.
        min_heigth (number, optional): altura minima de las observaciones. Expresado como porcentaje. Si no se especifica se calcula como el 10% del alto.
        min_width (number, optional): ancho minimo de las observaciones. Expresado como porcentaje. Si no se especifica se calcula como el 10% del ancho.
    """
    
    angle_base = rng.uniform(-15, 15)
    source_y = 0
    alto_disponible = alto
    ancho_disponible = ancho
    min_heigth_px = min_heigth * alto
    min_width_px = min_width * ancho
    
    observations_limits = []
    random_iter = 0
    while (alto_disponible > 0 and random_iter < 0.5 and alto_disponible <= alto):
        angle = angle_base + rng.integers(-3, 3)
        angle_rad = math.radians(angle)
        angle_cos = math.cos(angle_rad)
        angle_sin = math.sin(angle_rad)
        # Centro en x de la observacion
        coor_x = rng.integers(0.25*ancho_disponible, 0.75*ancho_disponible)
        # Ancho de la observacion
        diff_extremos = min(coor_x, ancho_disponible - coor_x)
        if(min_width_px > diff_extremos):
            break
        ancho_general = rng.integers(min_width_px, diff_extremos) * 2
        # Ancho de la observacion rotada (solo linea)
        ancho_obs = angle_cos * ancho_general
        # Alto de la observacion rotada (solo linea)
        alto_obs = abs(angle_sin * ancho_general)
        if(alto_obs > alto_disponible):
            break
        # Centro en y de la observacion
        low = math.ceil(alto_disponible * 0.025 + alto_obs / 2)
        high = math.floor(alto_disponible * 0.975 - alto_obs / 2)
        if low >= high:
            break
        coor_y = rng.integers(low, high)
        # Diferencia vertical entre el extremo central derecho de la observacion y su pico superior derecho
        min_heigth_px_vertical = angle_cos * min_heigth_px
        dif_extremo_disp_y = min(coor_y - alto_obs/2, alto_disponible - (coor_y + alto_obs/2))
        if(min_heigth_px_vertical > dif_extremo_disp_y):
            break
        dif_extremo_disp_y = rng.uniform(
            min_heigth_px_vertical/2, 
            dif_extremo_disp_y
        )
        # Alto de la observacion sin rotar
        apertura = (dif_extremo_disp_y / angle_cos) * 2
        # Pendientes de los ejes de la observacion        
        ux = angle_cos
        uy = angle_sin
        vx = -angle_sin
        vy =  angle_cos
        # Precomputo
        half_len = ancho_general / 2
        half_wid = apertura / 2
        
        # Especificar la observacion con sus limites
        new_observation = ObservationLimit(
                angle=angle,
                alto=apertura,
                ancho=ancho_general,
                x_center=coor_x,
                y_center=source_y+coor_y,
                x1=coor_x - ux*half_len - vx*half_wid,
                y1=coor_y - uy*half_len - vy*half_wid + source_y,
                x2=coor_x + ux*half_len - vx*half_wid,
                y2=coor_y + uy*half_len - vy*half_wid + source_y,
                x3=coor_x + ux*half_len + vx*half_wid,
                y3=coor_y + uy*half_len + vy*half_wid + source_y,
                x4=coor_x - ux*half_len + vx*half_wid,
                y4=coor_y - uy*half_len + vy*half_wid + source_y
            )
        observations_limits.append(new_observation)
        
        # Minimos y maximos de la observacion para definir el espacio disponible
        obs_min_y = min(new_observation.y1, new_observation.y2, new_observation.y3, new_observation.y4) - source_y
        obs_max_y = max(new_observation.y1, new_observation.y2, new_observation.y3, new_observation.y4) - source_y
        
        # Calcular el espacio superior e inferior a la observacion para definir donde se ubica la fuente y el espacio disponible para la siguiente observacion
        espacio_superior = obs_min_y
        espacio_inferior = alto_disponible - obs_max_y
        if(espacio_superior > espacio_inferior):
            # El espacio superior es mayor, solo se ajusta el alto disponible y se mantiene la fuente en el mismo lugar
            alto_disponible = obs_min_y
        else:
            # El espacio inferior es mayor, se mueve la fuente hacia abajo y se ajusta el alto disponible
            source_y = source_y + obs_max_y
            alto_disponible = alto_disponible - obs_max_y
            
        random_iter = rng.random()
    
    # print("Random final para definir observaciones:", random_iter)
    # print("Alto disponible luego de definir observaciones:", alto_disponible)
    # print(f"Se generaron {len(observations_limits)} observaciones con los siguientes limites:")
    return observations_limits
