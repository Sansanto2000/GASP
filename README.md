# GASP

GASP (Generator for Astronomical Spectroscopic Plates) es un conjunto de herramientas para la generación de imágenes sintéticas de escaneos de placas espectroscópicas.

De cada imagen generada se provee tanto la imagen como la información de los elementos que contiene haciendo las imágenes adecuadas para flujos de trabajo con modelos de visión por computadora como YOLO.

![Imagen sintética de un escaneo de una placa espectroscópica con 2 observaciones.](assets/exampleGeneration3.jpg)

![Imagen sintética de un escaneo de una placa espectroscópica con 1 observación. En azul los limites que delimitan la posición de la observación generada.](assets/exampleGeneration1.jpg)

![Imagen sintética de un escaneo de una placa espectroscópica con 4 observaciones. En azul los limites que delimitan la posición de cada una de las observaciones generadas.](assets/exampleGeneration2.jpg)

## Formato de etiquetas

El generador produce los siguientes formatos de etiquetas

|Formato    |Anatomia                                           |Descripcion    |
|:---------:|:-------------------------------------------------:|:--------------|
|YOLO       |`<class> <x> <y> <w> <h>`                          |`class` toma valor entero respecto al indice de la clase del *Bounding Box*. Los demas toman valores porcentuales [0-1]. `x` e `y` las cordenadas del centro de la observacion. E `w` y `h` son el ancho y alto respectivamente. |
|OBB        |`<class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`  |`class` toma valor entero respecto al indice de la clase del *Bounding Box*. Los demas toman valores porcentuales [0-1]. `x1`, `y1`, `x2`, `y2`, `x3`, `y3`, `x4` e `y4` corresponden a las coordenadas de cada esquina de la caja delimitadora.   |

Las etiquetas producidas por el generador están en formato YOLO. En esquema *rel_xywh*, osea 5 datos, el primero entero, los demás floats normalizados como valores entre 0 y 1:

# Tipos de etiquetas

El generador produce imagenes etiquetadas respecto a distintas clases. El que etiquetas se generan se puede especificar via parametros, a continuacion se muestra un ejemplo de cada caso.

|Clase          |Descripcion                            |Imagen |
|:-------------:|:--------------------------------------|:-----:|
|`observacion`  |Observacion espectroscopica.           |![Observacion espectroscopica.](assets/...)|
|`science`      |Espectro de ciencia.                   |![Espectro de ciencia.](assets/...)|
|`lamp`         |Espectro de lampara de comparacion.    |![Espectro de lampara de comparacion.](assets/...)|

Cuando se selecciona mas de una clase entonces 


# Entorno virtual

Se recomienda usar *uv* para la administración del entorno virtual.

## Generar

En `generators\spectrumLabeledSequence` se encuentra un generador compatible con la librería TensorFlow. El archivo `generator_use_example.py` muestra un ejemplo de como usarla para generar y almacenar archivos, este puede ser usado como se muestra a continuación. 

```
uv run generator_save_images_example.py
```

Toda la herramienta es **importable como una libreria**.