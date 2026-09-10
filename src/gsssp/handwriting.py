"""Anotaciones manuscritas junto a las observaciones, a partir de EMNIST Letters."""
import cv2
import numpy as np
from numpy.typing import NDArray

_LETTERS = None
_LETTERS_BY_LABEL = None  # {1..26: NDArray de indices en _LETTERS con esa letra}

def _load_letters():
    """Carga y cachea a nivel de modulo los glifos de EMNIST Letters, agrupados por letra.

    Son datos inmutables, no estado que afecte la reproducibilidad: la seleccion de que
    instancia de cada letra usar en cada anotacion se hace despues, indexando estos
    arreglos con el rng de la placa. Se cargan una sola vez por proceso.

    Agrupar por letra (1=A .. 26=Z, la convencion de EMNIST) es lo que permite pedir
    letras consecutivas del alfabeto (g, h, i, ...) en vez de solo indices consecutivos
    del dataset, que no esta ordenado alfabeticamente.

    Requiere el dataset descargado (`tfds.load('emnist/letters', split='train')` corrido
    una vez con internet, se cachea en ~/tensorflow_datasets).

    Return:
    - {NDArray[np.uint8]}: glifos con forma (N, 28, 28), orientados en forma legible.
    - {dict[int, NDArray[int]]}: indices en el arreglo de glifos, por letra (1-26).
    """
    global _LETTERS, _LETTERS_BY_LABEL
    if _LETTERS is None:
        import tensorflow_datasets as tfds
        ds = tfds.load("emnist/letters", split="train", as_supervised=True)
        images, labels_list = [], []
        for img, label in ds.as_numpy_iterator():
            # EMNIST viene rotado: transponer alcanza para la orientacion legible
            # (confirmado contra las clases reales del dataset, no solo contra la nota
            # de la documentacion).
            images.append(img[:, :, 0].T)
            labels_list.append(int(label))
        _LETTERS = np.stack(images)
        labels_arr = np.array(labels_list)
        _LETTERS_BY_LABEL = {
            letra: np.flatnonzero(labels_arr == letra) for letra in range(1, 27)
        }
    return _LETTERS, _LETTERS_BY_LABEL

def _compose_glyph(img, glyph, x0, y0, opacity, ink_max):
    """Compone un glifo sobre img en (x0, y0) por alfa, recortando contra el canvas.

    `opacity` y `ink_max` controlan cosas distintas: `opacity` es cuanto se ve el
    fondo a traves del trazo (que tan solido queda), `ink_max` es el techo de brillo
    de la tinta (para que un trazo bien opaco no sea blanco puro, sino gris).
    """
    h, w = img.shape[:2]
    gh, gw = glyph.shape
    y1, x1 = max(0, y0), max(0, x0)
    y2, x2 = min(h, y0 + gh), min(w, x0 + gw)
    if y2 <= y1 or x2 <= x1:
        return
    crop = glyph[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0].astype(np.float32)
    alpha = (crop / 255.0) * opacity
    ink = crop * (ink_max / 255.0)
    region = img[y1:y2, x1:x2]
    region[:] = region * (1 - alpha) + ink * alpha

def add_observation_annotations(
    img: NDArray[np.uint8],
    posiciones,
    obs_width: float,
    obs_heigth: float,
    angle: float = 0,
    *, rng: np.random.Generator = None
) -> NDArray[np.uint8]:
    """Agrega una letra manuscrita junto a cada observacion de la placa.

    Simula la anotacion a mano (identificador, fecha, numero de placa) que suele
    aparecer al costado de las observaciones en un escaneo real. Es un distractor: no
    se etiqueta como clase, solo busca que el modelo no confunda escritura real con una
    observacion.

    El lado (izquierda o derecha), el tamaño de letra y la distancia a la observacion se
    sortean una sola vez para toda la placa, porque todas las observaciones de una misma
    placa comparten ancho y alto (`obs_width`, `obs_heigth` son unicos por placa, no por
    observacion) — asi la anotacion queda alineada en una columna consistente, como la
    escribiria una sola persona. La distancia objetivo se recorta por observacion solo si
    esa observacion en particular no tiene espacio suficiente (esta mas cerca del borde
    de la placa que las demas), no como variacion porque si.

    Compone por alfa en vez de con np.maximum (que es como el resto del pipeline funde
    la observacion), porque el alfa da control sobre que tan marcado se ve el trazo:
    `fondo*(1-a) + tinta*a`, con `a` derivado de la intensidad del glifo.

    Se llama antes de agregar el borde de la placa (`add_plate_edge`) para que, si la
    placa tiene borde, este quede pintado por encima: una anotacion nunca puede quedar
    sobre la zona de borde, que representa el limite fisico de la emulsion.

    Cada glifo se desenfoca levemente y se compone con opacidad menor a 1 para que no
    quede con el trazo perfectamente nitido de EMNIST (un dataset escaneado a proposito
    en alta resolucion): una anotacion real, vista a la escala de la placa, se ve mas
    parecida a esto que a un glifo de dataset sin alterar.

    Se rota cada glifo el mismo `angle` que la observacion correspondiente, con la misma
    convencion de signo que `draw_observation` (angulo positivo = la observacion baja
    hacia la derecha; `cv2.getRotationMatrix2D` usa el criterio opuesto, por eso se la
    llama con `-angle`), para que la anotacion se vea escrita a lo largo de la misma
    inclinacion que la observacion, no derecha contra un fondo inclinado.

    La referencia vertical de cada letra no es el centro geometrico de la observacion,
    sino el punto medio de su lado izquierdo o derecho (el que corresponda) ya rotado —
    la misma matriz `M = cv2.getRotationMatrix2D((x,y), -angle, 1)` que usa
    `draw_observation` para rotar los componentes. Si la observacion esta inclinada, ese
    punto no tiene la misma altura que el centro: usar el centro haria que la letra
    parezca demasiado arriba o abajo respecto del lateral real de la observacion.

    Puede haber superposicion parcial entre la letra y el espectro: el margen entre
    ambos a veces se sortea negativo a proposito (una persona anotando a mano no
    siempre deja un espacio limpio). Tambien puede terminar muy cerca del borde de la
    placa si la observacion ya estaba cerca: la distancia lateral no conoce el ancho
    de la placa, solo el de la observacion. Los pixeles fuera del canvas simplemente
    se recortan (`_compose_glyph`), sin error.

    Las letras de una misma placa son todas distintas y consecutivas del alfabeto (por
    ejemplo g, h, i, ...): se sortea una letra de arranque y se recorre el alfabeto
    desde ahi, en vez de sortear cada una de forma independiente. Da la vuelta al
    llegar a la Z.

    La distancia a la observacion tiene un piso minimo respecto del borde real de la
    placa (no solo de la observacion), salvo cuando toca superponerse al espectro: ahi
    la distancia se mide hacia el lado contrario (invade la observacion, se aleja del
    borde), asi que no hace falta el piso. Fuera de ese caso, la distancia objetivo
    sale de una mezcla de dos normales sorteada una sola vez por placa (la mayoria de
    las placas la anotacion queda cerca de la observacion, algunas bastante mas lejos)
    y se recorta por observacion solo si esa en particular no tiene espacio suficiente.

    Parametros:
    - img {NDArray[np.uint8]}: placa sobre la que componer, en escala de grises.
    - posiciones {list[dict]}: centros de las observaciones ya dibujadas, con claves
    "x" e "y" en pixeles (la misma lista que usa `generar_placa` para dibujarlas).
    - obs_width, obs_heigth {float}: ancho y alto nominal de las observaciones de esta
    placa (unico por placa).
    - angle {float}?: inclinacion en grados de las observaciones de esta placa (unica
    por placa, igual que `obs_width`/`obs_heigth`). Default 0.
    - rng {np.random.Generator}?: generador aleatorio a usar. Si no se pasa se crea uno
    sin semilla.

    Return:
    - {NDArray[np.uint8]}: la placa con las anotaciones compuestas.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not posiciones:
        return img

    letters, letters_by_label = _load_letters()
    h, w = img.shape[:2]
    prob_skip = 0.02
    prob_above = 0.02
    prob_overlap = 0.15
    border_margin = 0.03 * w  # piso minimo de distancia al borde real de la placa

    side = "left" if rng.random() < 0.5 else "right"
    gap_base = rng.uniform(0.08, 0.25) * obs_heigth
    edge_offset_x = -obs_width / 2 if side == "left" else obs_width / 2
    # Letra de arranque unica por placa: recorrer el alfabeto desde ahi (en vez de
    # sortear cada letra de forma independiente) garantiza que sean consecutivas y que
    # ninguna se repita dentro de la misma placa.
    start_letter = int(rng.integers(1, 27))
    # Tamaño unico por placa, igual que el resto de los parametros compartidos: todas
    # las anotaciones de una misma placa se ven escritas del mismo tamaño.
    char_height = max(1, int(obs_heigth * rng.uniform(0.35, 0.9)))
    # Distancia objetivo, tambien unica por placa: mezcla de dos normales en vez de
    # una uniforme (la mayoria de las placas la anotacion queda cerca, algunas bastante
    # mas lejos). Por observacion solo se recorta si esa en particular no tiene
    # espacio suficiente, no como variacion porque si.
    if rng.random() < 0.7:
        target_gap = rng.normal(gap_base, gap_base * 0.25)
    else:
        target_gap = rng.normal(gap_base * 2.5, gap_base * 0.5)
    target_gap = max(0.0, target_gap)

    for i, coor in enumerate(posiciones):
        if rng.random() < prob_skip:
            continue

        # Punto medio del lateral de la observacion (izquierdo o derecho, el que
        # corresponda), ya rotado: no el centro geometrico de la observacion.
        M = cv2.getRotationMatrix2D((coor["x"], coor["y"]), -angle, 1.0)
        edge_point = M @ np.array([coor["x"] + edge_offset_x, coor["y"], 1.0])
        ref_x, ref_y = edge_point[0], edge_point[1]

        letra = ((start_letter - 1 + i) % 26) + 1
        indices_letra = letters_by_label[letra]
        glyph = letters[indices_letra[int(rng.integers(0, len(indices_letra)))]]
        glyph = cv2.resize(glyph, (char_height, char_height), interpolation=cv2.INTER_LINEAR)

        if angle != 0:
            center = (char_height / 2, char_height / 2)
            M_glyph = cv2.getRotationMatrix2D(center, -angle, 1.0)
            glyph = cv2.warpAffine(glyph, M_glyph, (char_height, char_height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Desenfoque leve: sin esto el trazo queda con la nitidez de un dataset escaneado
        # a proposito, no con la de una anotacion vista a la escala de la placa.
        blur_ksize = max(3, (char_height // 12) | 1)
        glyph = cv2.GaussianBlur(glyph, (blur_ksize, blur_ksize), 0)

        # Opacidad aleatoria por letra: una anotacion real no siempre queda igual de
        # marcada (presion del trazo, tipo de lapiz). ink_max techa el brillo maximo
        # del trazo para que, aun con opacidad alta, no quede blanco puro — pero lo
        # bastante alto para no perderse contra el fondo.
        opacity = rng.uniform(0.6, 0.95)
        ink_max = rng.uniform(190, 235)

        if rng.random() < prob_overlap:
            # Invade el espectro a proposito: se mide hacia el lado de la observacion,
            # no hacia el borde de la placa, asi que no hace falta el piso de abajo.
            gap = -rng.uniform(0.2, 0.8) * char_height
        else:
            # La misma distancia objetivo de toda la placa, recortada solo si esta
            # observacion en particular no tiene espacio (esta mas cerca del borde
            # real que las demas).
            available = ref_x if side == "left" else (w - ref_x)
            max_gap = max(0, available - char_height - border_margin)
            gap = min(target_gap, max_gap)
        if side == "left":
            x0 = int(round(ref_x - gap - char_height))
        else:
            x0 = int(round(ref_x + gap))

        # Centrada en el lateral de la observacion, con variacion vertical leve; con
        # cierta probabilidad, se corre hacia arriba (desde el mismo costado).
        if rng.random() < prob_above:
            y_center = ref_y - char_height * rng.uniform(0.1, 0.6)
        else:
            y_center = ref_y + rng.uniform(-0.05, 0.05) * obs_heigth
        y0 = int(round(y_center - char_height / 2))

        _compose_glyph(img, glyph, x0, y0, opacity, ink_max)

    return img
