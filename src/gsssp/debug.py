"""Utilidades de visualizacion para depurar la geometria de observaciones."""
import math

import cv2
import numpy as np

def visualize_observations(
    observations,
    img_width=800,
    img_height=600,
    output_path="observation_debug.png"
):
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for obs in observations:
        # Centro
        cx = int(obs.x_center)
        cy = int(obs.y_center)
        center = (cx, cy)
        cv2.circle(img, center, 5, (0, 0, 255), -1)

        angle_rad = math.radians(obs.angle)

        # Eje de ancho
        dx = math.cos(angle_rad) * obs.ancho / 2
        dy = math.sin(angle_rad) * obs.ancho / 2

        width_start = (int(cx - dx), int(cy - dy))
        width_end   = (int(cx + dx), int(cy + dy))

        # Eje de alto (perpendicular)
        px = -math.sin(angle_rad) * obs.alto / 2
        py =  math.cos(angle_rad) * obs.alto / 2

        height_start = (int(cx - px), int(cy - py))
        height_end   = (int(cx + px), int(cy + py))

        # Dibujar ejes
        cv2.line(img, width_start, width_end, (0, 255, 0), 2)   # ancho = verde
        cv2.line(img, height_start, height_end, (255, 0, 0), 2) # alto = azul

        # Esquinas
        p1 = (int(obs.x1), int(obs.y1))
        p2 = (int(obs.x2), int(obs.y2))
        p3 = (int(obs.x3), int(obs.y3))
        p4 = (int(obs.x4), int(obs.y4))

        # Dibujar cada esquina con un color distinto
        cv2.circle(img, p1, 6, (255, 0, 0), -1)     # Azul
        cv2.circle(img, p2, 6, (0, 255, 0), -1)     # Verde
        cv2.circle(img, p3, 6, (0, 255, 255), -1)   # Amarillo
        cv2.circle(img, p4, 6, (255, 0, 255), -1)   # Magenta

        # Etiquetas
        cv2.putText(img, "P1", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, "P2", p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, "P3", p3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img, "P4", p4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Contorno del OBB
        pts = np.array([p1, p2, p3, p4], dtype=np.int32)
        cv2.polylines(
            img,
            [pts],
            isClosed=True,
            color=(255, 255, 255),
            thickness=2
        )

    cv2.imwrite(output_path, img)
    print(f"Imagen guardada en: {output_path}")
