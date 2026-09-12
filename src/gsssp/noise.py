"""Ruido de escaneo y bordes de placa."""
from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

def add_background_field(
    img: NDArray[np.uint8],
    amplitude: float,
    grid: int = 8,
    *, rng: np.random.Generator = None
) -> NDArray[np.uint8]:
    """Aplica un campo de iluminacion de baja frecuencia, multiplicativo.

    Simula iluminacion despareja, vineteado optico y velo quimico desigual: efectos de
    baja frecuencia y bidireccionales que una placa fotografica escaneada real presenta
    y que el ruido de alta frecuencia de add_realistic_noise no cubre. Se aplica sobre
    el canvas antes de dibujar las observaciones, para que el gradiente las afecte
    tambien a ellas, como pasa fisicamente cuando la iluminacion del escaner es
    despareja.

    Se genera ruido en una grilla chica (grid x grid) y se escala al tamaño de la
    imagen con interpolacion bicubica, en vez de ruido por pixel, porque es lo que
    da la variacion de gran escala sin necesidad de otra dependencia.

    Parametros:
    - amplitude {float}: amplitud del campo. 0 no aplica ningun efecto (el canvas
    queda igual que antes). Valores mas altos oscurecen o aclaran zonas grandes de
    la placa de forma mas marcada.
    - grid {int}?: tamaño de la grilla base antes de interpolar. Default 8.
    - rng {np.random.Generator}?: generador aleatorio a usar. Si no se pasa se crea
    uno sin semilla.
    """
    if amplitude <= 0:
        return img

    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape[:2]
    small = rng.normal(1.0, amplitude, (grid, grid)).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)

def add_realistic_noise(
    img: NDArray[np.uint8],
    gaussian_std: float = 10.0,
    band_intensity: float = 5.0,
    speck_count: int = 10,
    speck_size: int = 3,
    blur_ksize: int = 3,
    violin_line_count: int = 0,
    violin_intensity = 0.3,
    violin_length_range = (0.05, 0.7),
    grain_std: float = 0.0,
    scratch_line_count: int = 0,
    scratch_intensity = 0.5,
    scratch_length_range = (0.02, 0.8),
    hair_line_count: int = 0,
    hair_intensity = 0.5,
    hair_length_range = (0.005, 0.05),
    *,
    valid_mask: NDArray[np.bool_] = None,
    rng: np.random.Generator = None
) -> NDArray[np.uint8]:
    """Añadir ruido realista a una imagen.

    Trabaja en escala de grises: recibe y devuelve un arreglo (alto, ancho). La expansion
    a tres canales se hace una sola vez al final del pipeline, no en cada etapa.

    Parametros:
    - gaussian_std {float}?: ruido gaussiano. Simula imperfecciones naturales del sensor 
    o de la pelicula fotografica. Se basa en una distribucion normal o gaussiana. Default 10.0.
    - band_intensity {float}?: ruido de banda (horizontal o vertical). Default 5.0.
    - speck_count {int}?: cantidad de manchas de impuresa a simular.
    - speck_size {int}?: tamaño maximo de mancha de impuresa. Default 3.
    - blur_ksize {int}?: tamaño del kernel para desenfoque gaussiano. Debe ser impar, con 0 
    u otro valor invalido no se aplica ningun desenfoque. Default 3.
    - violin_line_count {int}?: cantidad de manchas alargadas tipo "violín" a simular. Default 0.
    - violin_intensity {float}?: intensidad de las manchas alargadas tipo "violín". Default 0.3.
    - violin_length_range {Tuple[float, float]}?: rango porcentual de longitud de las manchas
    alargadas tipo "violín". Default (0.05, 0.7).
    - grain_std {float}?: intensidad del grano de emulsion, ruido espacialmente
    correlacionado (a diferencia de gaussian_std, que es ruido blanco por pixel). 0 no
    aplica ningun efecto. Default 0.0.
    - scratch_line_count {int}?: cantidad de rayas finas de manipulacion (rayones
    diagonales que cruzan el fondo) a simular. Default 0.
    - scratch_intensity {float}?: intensidad de las rayas finas de manipulacion. Default 0.5.
    - scratch_length_range {Tuple[float, float]}?: rango porcentual de longitud de las
    rayas finas de manipulacion, relativo a la diagonal de la imagen. Default (0.02, 0.8).
    - hair_line_count {int}?: cantidad de marcas curvas aisladas tipo pelo o fibra a
    simular. Default 0.
    - hair_intensity {float}?: intensidad de las marcas curvas tipo pelo o fibra. Default 0.5.
    - hair_length_range {Tuple[float, float]}?: rango porcentual de longitud de las
    marcas curvas tipo pelo o fibra, relativo a la diagonal de la imagen. Default
    (0.005, 0.05): mucho mas cortas que las rayas de manipulacion, consistente con un
    pelo o fibra sobre el negativo o el vidrio del escaner, no con un rayon que cruza
    la placa.
    - valid_mask {NDArray[np.bool_]}?: mascara (alto, ancho) del lado valido de un borde
    de placa ya dibujado (ver `add_plate_edge`), si lo hay. Las rayas de manipulacion no
    se dibujan del lado False: una raya real esta sobre la emulsion, y del otro lado del
    borde fisico no hay emulsion que rayar. None (default) no restringe nada.
    - rng {np.random.Generator}?: generador aleatorio a usar. Si no se pasa se crea uno
    sin semilla. Recibirlo permite que el resultado sea reproducible y seguro entre hilos.
    """
    if rng is None:
        rng = np.random.default_rng()

    img_noisy = img.astype(np.float32)

    # 1. Ruido gaussiano (general)
    noise = rng.normal(0, gaussian_std, img.shape)
    img_noisy += noise

    # 1b. Grano de emulsion: mismo ruido gaussiano pero desenfocado antes de sumarlo, para
    # que quede espacialmente correlacionado en vez de independiente por pixel.
    if grain_std > 0:
        grain = rng.normal(0, grain_std, img.shape).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (5, 5), 0)
        grain *= grain_std / (grain.std() + 1e-6)
        img_noisy += grain

    # 2. Ruido en bandas horizontales (tiras verticales o líneas horizontales)
    band = rng.normal(0, band_intensity, (img.shape[0], 1))
    img_noisy += band

    # 3. Puntos blancos o manchas (tipo polvo o defecto)
    for _ in range(speck_count):
        cx = rng.integers(0, img.shape[1])
        cy = rng.integers(0, img.shape[0])
        radius = 1 if speck_size <= 1 else rng.integers(1, speck_size)
        color = rng.integers(150, 255)  # blanco sucio
        cv2.circle(img_noisy, (int(cx), int(cy)), int(radius), int(color), cv2.FILLED)

    # 4. Manchas alargadas
    violin_sigma: float = 3.0
    h, w = img.shape[:2]
    # Ejes como columna y fila: al combinarlos broadcastean a (h, w) sin materializar
    # dos grillas completas por mancha, como hacia np.meshgrid.
    eje_y = np.arange(h)[:, None]
    eje_x = np.arange(w)[None, :]
    for _ in range(violin_line_count):
        violin_length_ratio: float = rng.uniform(*violin_length_range)

        # Centro x, y aleatorio
        y0 = rng.integers(0, h)
        # Centro horizontal (evita bordes)
        x0 = rng.integers(int(w * 0.1), int(w * 0.9)+1)

        # Largo horizontal (no llega a bordes)
        L = int(w * violin_length_ratio)
        sigma_x = L / 3

        # Gaussiana 2D estirada horizontalmente
        gaussian_2d = 255 * violin_intensity * np.exp(
            -(
                ((eje_y - y0) ** 2) / (2 * violin_sigma ** 2) +
                ((eje_x - x0) ** 2) / (2 * sigma_x ** 2)
            )
        )
        # Aplicar
        img_noisy += gaussian_2d

    # 5. Rayas finas de manipulacion: rayones diagonales, nitidos (a diferencia de las
    # manchas "violín", que son difusas), que se cruzan entre si sobre el fondo. El angulo
    # se mantiene lejos tanto de la horizontal como de la vertical para no confundirse con
    # el borde de una observacion o con una linea espectral.
    scratch_angle_deg_range = (15.0, 75.0)
    diag = float(np.hypot(h, w))
    # Las rayas se dibujan siempre directo sobre el fondo real (mismo blend que sin
    # mascara: cv2.line reemplaza el pixel, no lo suma). Si hay un lado de corte, se
    # guarda esa zona antes de dibujar y se restaura despues en un solo paso, en vez
    # de recortar raya por raya: mas barato (una copia del tamaño de la zona de
    # corte, no del canvas completo, una sola vez) y evita que dibujar sobre un
    # lienzo en cero y sumar duplique el brillo donde el fondo ya era claro.
    invalid_backup = img_noisy[~valid_mask].copy() if valid_mask is not None else None
    for _ in range(scratch_line_count):
        length = diag * rng.uniform(*scratch_length_range)
        angle_deg = rng.uniform(*scratch_angle_deg_range) * rng.choice([-1, 1])
        angle_rad = np.deg2rad(angle_deg)
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        dx = np.cos(angle_rad) * length / 2
        dy = np.sin(angle_rad) * length / 2
        pt1 = (int(cx - dx), int(cy - dy))
        pt2 = (int(cx + dx), int(cy + dy))
        cv2.line(img_noisy, pt1, pt2, 255 * scratch_intensity, thickness=1, lineType=cv2.LINE_AA)
    if invalid_backup is not None:
        img_noisy[~valid_mask] = invalid_backup

    # 6. Marca curva aislada tipo pelo o fibra: consistente con un pelo o fibra que
    # quedo sobre el negativo o el vidrio del escaner al momento de digitalizar, a
    # diferencia de las rayas de manipulacion del paso anterior (rectas, en red, de
    # origen en la emulsion). Se dibuja como una polilinea de pocos segmentos, con el
    # angulo variando levemente de un segmento al siguiente en vez de mantenerse fijo,
    # lo que da una curva irregular corta en vez de un eje recto. El grosor arranca
    # ancho y algo circular en la raiz (folicular) y se afina hacia la punta, en vez de
    # mantenerse constante: es lo que se ve en un pelo o fibra real sobre el escaneo.
    hair_segment_count_range = (3, 7)
    hair_segment_angle_jitter_deg = 30.0
    hair_root_thickness_range = (2, 4)
    hair_tip_thickness = 1
    for _ in range(hair_line_count):
        length = diag * rng.uniform(*hair_length_range)
        n_segments = int(rng.integers(*hair_segment_count_range))
        segment_length = length / n_segments
        angle_deg = rng.uniform(0, 360)
        color = 255 * hair_intensity
        root_thickness = int(rng.integers(*hair_root_thickness_range))
        x, y = float(rng.integers(0, w)), float(rng.integers(0, h))
        cv2.circle(
            img_noisy, (int(round(x)), int(round(y))), root_thickness // 2 + 1,
            color, cv2.FILLED, lineType=cv2.LINE_AA,
        )
        for i in range(n_segments):
            angle_deg += rng.uniform(-hair_segment_angle_jitter_deg, hair_segment_angle_jitter_deg)
            x_next = x + segment_length * np.cos(np.deg2rad(angle_deg))
            y_next = y + segment_length * np.sin(np.deg2rad(angle_deg))
            thickness = round(
                root_thickness + (hair_tip_thickness - root_thickness) * (i + 1) / n_segments
            )
            cv2.line(
                img_noisy, (int(round(x)), int(round(y))), (int(round(x_next)), int(round(y_next))),
                color, thickness=max(1, thickness), lineType=cv2.LINE_AA,
            )
            x, y = x_next, y_next

    # 7. Desenfoque suave (simula ópticas imperfectas)
    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        img_noisy = cv2.GaussianBlur(img_noisy, (blur_ksize, blur_ksize), 0)


    # Clip y convertir de vuelta a uint8
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    return img_noisy

class Position(Enum):
    RIGHT = 0
    LEFT = 1
    TOP = 2
    BOTTOM = 3

def _add_edge_glow(img, interior, max_thickness, *, rng: np.random.Generator):
    """Agrega un brillo que decae desde el borde hacia adentro de la placa.

    Simula el filo mas revelado de la emulsion justo junto al borde fisico: una
    franja clara pegada al limite que se atenua rapido a medida que se aleja hacia
    el interior (caida tipo exponencial desde el borde), en vez de que el borde
    corte de golpe al color de fondo parejo. Hacia afuera del borde no cambia nada:
    ese lado ya quedo resuelto por el relleno parejo de `add_plate_edge`.

    La caida no es una exponencial prolija: la distancia al borde se perturba con
    ruido de baja frecuencia (revelado disparejo) y el brillo resultante ademas
    lleva grano fino pixel a pixel, para que no se vea como un degradado calculado
    sino como una franja irregular, igual que el resto del ruido del pipeline.

    Args:
        img (NDArray[np.uint8]): imagen ya con el borde (fillPoly) aplicado.
        interior (NDArray[np.bool_]): mascara del lado de la placa (True) contra
        el lado de corte (False), calculada una sola vez en `add_plate_edge`.
        max_thickness (int): grosor maximo disponible en este lado, usado para
        escalar que tan lejos hacia adentro llega el brillo.
        rng (np.random.Generator): generador aleatorio a usar.

    Returns:
        NDArray[np.uint8]: imagen con el brillo agregado.
    """
    h, w = interior.shape

    # Distancia (en pixeles) de cada punto interior al borde de corte mas cercano.
    dist = cv2.distanceTransform((interior.astype(np.uint8)) * 255, cv2.DIST_L2, 5)

    peak = rng.uniform(80.0, 200.0)
    decay_length = rng.uniform(0.015, 0.12) * max(max_thickness, 3)

    # Revelado disparejo: se perturba la distancia con ruido suavizado (a la misma
    # escala que la caida) antes de la exponencial, asi el limite del brillo no
    # sigue la curva del borde de forma pareja sino con entrantes y salientes.
    roughness = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    roughness = cv2.GaussianBlur(roughness, (0, 0), sigmaX=max(decay_length * 0.5, 1.0))
    roughness *= decay_length * rng.uniform(0.4, 0.9)
    dist_rough = np.clip(dist + roughness, 0, None)

    glow = peak * np.exp(-dist_rough / decay_length)

    # Grano fino superpuesto, proporcional al brillo local: sin esto la franja
    # queda como un degradado calculado en vez de una mancha fotografica.
    grain = rng.normal(1.0, 0.3, (h, w)).astype(np.float32)
    glow *= grain

    glow[~interior] = 0.0

    return np.clip(img.astype(np.float32) + glow, 0, 255).astype(np.uint8)

def add_plate_edge(
    img, edges, position:Position, *,
    prob_glow_edge: float = 0.5,
    rng:np.random.Generator = None,
):
    """Agrega un borde a la placa basado en los limites de las etiquetas.

    Trabaja en escala de grises: recibe y devuelve un arreglo (alto, ancho).

    El borde es curvo/irregular, nunca una linea recta: se arma mezclando una leve
    inclinacion general con arcos de radio aleatorio a lo largo del lado, para imitar
    como la emulsion de una placa real se contrae de forma desigual cerca del borde.
    Independientemente de la forma del borde, con probabilidad `prob_glow_edge` se le
    suma ademas un brillo pegado al limite que decae hacia el interior (ver
    `_add_edge_glow`).

    Tambien devuelve la mascara del lado de la placa (contra el lado de corte),
    para que quien llama pueda usarla despues y evitar que un efecto de ruido
    dibujado mas tarde (una raya de manipulacion, por ejemplo) cruce el borde
    hacia la zona de corte.

    Args:
        img (NDArray[np.uint8]): imagen a modificar.
        edges (tupla): (x_min, x_max, y_min, y_max) limites en pixeles donde
        se mueven las etiquetas.
        position (Position): lado de la placa donde agregar el borde.
        prob_glow_edge (float, optional): probabilidad de sumar el brillo que decae
        desde el borde hacia adentro. Default 0.5.
        rng (np.random.Generator, optional): generador aleatorio a usar. Si no se pasa
        se crea uno sin semilla.

    Returns:
        NDArray[np.uint8]: imagen con el borde agregado.
        NDArray[np.bool_]: mascara (alto, ancho), True del lado de la placa y
        False del lado de corte.
    """

    if rng is None:
        rng = np.random.default_rng()

    h, w = img.shape[:2]
    x_min, x_max, y_min, y_max = edges
    margin = 0.5
    # color del fondo "de atrás"
    gray = int(rng.integers(50, 156))
    bg_color = gray
    # inclinacion general leve del borde, de un extremo del lado al otro
    angle_noise = int(min(w, h) * 0.02)
    shift = int(rng.integers(-angle_noise, angle_noise + 1))

    # Cuantos arcos se mezclan a lo largo del lado y que tan marcados son. Son
    # constantes internas de forma, en la misma linea que hair_segment_count_range
    # o scratch_angle_deg_range: describen el "aspecto" del efecto, no algo que haga
    # falta barrer desde el generador.
    bump_count_range = (1, 4)
    bump_amplitude_range = (0.2, 0.4)  # fraccion de max_thickness
    bump_radius_range = (0.2, 0.4)  # fraccion de la longitud del lado

    def curved_profile(length, max_thickness):
        """Perfil de grosor a lo largo del lado: recta base + arcos gaussianos.

        Cada arco es una campana centrada en un punto aleatorio del lado, con un
        radio (ancho) y una amplitud (alto, positiva o negativa) aleatorios: hace
        de arco de radio variable sin necesidad de resolver la geometria de un
        arco de circunferencia real, que para este uso (variar el grosor del
        borde) da el mismo tipo de curva suave. La cantidad de arcos (1 a 4) no
        esta restringida: lo que se acota es su amplitud, para que la curva
        pueda ir y venir varias veces sin que cada vaiven sea muy profundo.

        El recorte a [0, max_thickness] se suaviza con un desenfoque liviano
        despues del clip: eso redondea el codo filoso que deja un np.clip duro
        justo donde un arco se pasa del margen disponible, sin comprimir el
        resto del perfil (que un suavizado tipo tanh sobre todo el rango si
        hace, achatando tambien las zonas que ya entraban sin problema).
        """
        base = int(rng.integers(0, max_thickness + 1))
        t = np.linspace(0.0, 1.0, length)
        profile = base + shift * t
        n_bumps = int(rng.integers(bump_count_range[0], bump_count_range[1] + 1))
        for _ in range(n_bumps):
            center = rng.uniform(0.0, 1.0)
            radius = rng.uniform(*bump_radius_range)
            amplitude = rng.uniform(*bump_amplitude_range) * max_thickness
            amplitude *= rng.choice([-1, 1])
            profile = profile + amplitude * np.exp(-((t - center) ** 2) / (2 * radius ** 2))
        clipped = np.clip(profile, 0, max_thickness)
        smoothing_sigma = max(length * 0.008, 2.0)
        clipped = cv2.GaussianBlur(
            clipped.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=smoothing_sigma
        ).flatten()
        return np.clip(clipped, 0, max_thickness)

    match position:
        case Position.RIGHT:
            max_thickness = max(1, int((w - x_max) * (1 - margin)))
            ys = np.arange(h)
            profile = curved_profile(h, max_thickness)
            curve = np.column_stack([w - profile, ys])
            pts = np.vstack([curve, [w, h], [w, 0]])
        case Position.LEFT:
            max_thickness = max(1, int(x_min * (1 - margin)))
            ys = np.arange(h)
            profile = curved_profile(h, max_thickness)
            curve = np.column_stack([profile, ys])
            pts = np.vstack([curve, [0, h], [0, 0]])
        case Position.TOP:
            max_thickness = max(1, int(y_min * (1 - margin)))
            xs = np.arange(w)
            profile = curved_profile(w, max_thickness)
            curve = np.column_stack([xs, profile])
            pts = np.vstack([[0, 0], [w, 0], curve[::-1]])
        case Position.BOTTOM:
            max_thickness = max(1, int((h - y_max) * (1 - margin)))
            xs = np.arange(w)
            profile = curved_profile(w, max_thickness)
            curve = np.column_stack([xs, h - profile])
            pts = np.vstack([[0, h], [w, h], curve[::-1]])

    pts = pts.astype(np.int32)
    cut_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(cut_mask, [pts], 255)
    valid_mask = cut_mask == 0

    cv2.fillPoly(img, [pts], bg_color)

    if rng.random() < prob_glow_edge:
        img = _add_edge_glow(img, valid_mask, max_thickness, rng=rng)

    return img, valid_mask
