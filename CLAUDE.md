# GASP — Generator for Astronomical Spectroscopic Plates

Generador de imágenes sintéticas de escaneos de placas espectroscópicas, con etiquetas
listas para entrenar modelos de detección (YOLO / Keras-CV). Se usa como **librería
importable** y también vía scripts de ejemplo.

> El repo se llama `GSSSP` en el filesystem y el paquete Python sigue siendo `gsssp`.
> El nombre público del proyecto es **GASP** (pasó por `G3SP` en el medio). No renombrar
> el paquete de forma oportunista: es un cambio propio, no algo que se cuela en otro commit.

## Estructura

```
src/gsssp/
  __init__.py                        API pública del paquete (__all__)
  observationArtist.py               núcleo de dibujado, ruido y etiquetas (~770 líneas)
  generators/
    spectrumLabeledSequence.py       keras.utils.Sequence que produce lotes (imagen, etiquetas)
src/test.py                          script manual de debug de define_observations_limits
generator_save_images_example.py     ejemplo: volcar un dataset a disco
assets/                              imágenes del README
```

`src/gsssp/generators/` **no tiene `__init__.py`** (namespace package implícito). Funciona,
pero tenerlo en cuenta si se toca el empaquetado.

Los imports son siempre **`gsssp.*`** (paquete instalado en modo editable), nunca
`src.gsssp.*`: esa segunda forma dependía del cwd y podía cargar el módulo dos veces.

### `observationArtist.py`

Dos generaciones de código conviven en este archivo:

- **Camino en producción** — `drawObservation()`, `spectral_function()`, `add_realistic_noise()`,
  `add_plate_edge()`, `edges_of_labels_relxywh()`, `labelDictToYolov11Format()`,
  `labelListToYolov11Format()`. Es lo que el generador usa hoy. Etiquetas AABB (`rel_xywh`).
- **Camino nuevo / WIP** — `ObservationLimit`, `ComponentLimit`, `define_observations_limits()`,
  `define_observation_components_limits()`, `visualize_observations()`. Apunta a etiquetas
  OBB (4 esquinas) y a componentes por clase (`observacion` / `science` / `lamp`).
  Todavía **no está conectado** al pipeline.

Estado conocido del WIP (no es deuda accidental, es trabajo a medio hacer):

- `spectrumLabeledSequence.__getitem__` ya llama `define_observations_limits()` con el
  `rng` del generador (`self.rng`), pero **el resultado todavía no se usa aguas abajo**:
  las observaciones se siguen posicionando con el código viejo basado en `random`.
  Conectarlo es el próximo paso del camino OBB.
- `define_observation_components_limits()` y `ObservationLimit.define_components_limits()`
  son `pass`.

### Convenciones de código

- Docstrings y comentarios en **español** (sin tildes en buena parte del código existente;
  seguir el estilo del archivo que se edita).
- `observationArtist.py` usa indentación de 4 espacios; `spectrumLabeledSequence.py` usa **2**.
  Respetar la del archivo, no unificar de paso.
- Aleatoriedad: el código viejo usa `random` / `np.random` globales; el código nuevo recibe
  un `np.random.Generator` (`rng`) explícito. Para código nuevo preferir `rng` inyectado.
  Hoy `SpectrumLabeledSequence` crea su `self.rng` sin semilla, así que el generador
  **todavía no es reproducible**; exponer la semilla es un cambio pendiente propio.
- Parámetros del generador: todos *keyword-only*, con rangos como tuplas `(min, max)`.
  Al agregar uno, sumarlo también al docstring de `__init__` y a `self.<nombre>`.

## Entorno

- Python **3.10** (`requires-python = ">=3.10,<3.11"`), gestionado con **uv**.
- TensorFlow 2.16.1 con dependencia condicional por plataforma: `tensorflow-macos` +
  `tensorflow-metal` en darwin, `tensorflow[and-cuda]` en linux. **No tocar ese bloque
  de `pyproject.toml` sin considerar ambas plataformas** — se desarrolla en macOS y se
  entrena en Linux con GPU.
- `uv.lock` está versionado: si cambian dependencias, el lock va **en el mismo commit**.

```bash
uv run generator_save_images_example.py
```

## Formatos de etiquetas

| Formato | Anatomía | Notas |
|---|---|---|
| YOLO (AABB) | `<class> <x> <y> <w> <h>` | class entero; resto normalizado [0-1], centro + tamaño. Lo que produce hoy el pipeline. |
| OBB | `<class> <x1> <y1> ... <x4> <y4>` | 4 esquinas normalizadas. Objetivo del camino WIP. |

Clases previstas: `observacion`, `science`, `lamp`.

---

# Git

## Mensajes de commit

```
<feat|fix>: [ID-XXX] <emoji> descripción corta
```

- `[ID-XXX]` solo si hay un Issue concreto; si no, se omite por completo.
- Emoji en formato shortcode (`:sparkles:`, `:bug:`, `:memo:`, `:wrench:`, `:fire:`,
  `:pushpin:`, `:package:`, `:truck:`), consistente con el historial.
- Un commit = **un solo propósito**, y chico. Nada de arrastrar reformateos, renombres
  de proyecto o ajustes de parámetros de prueba dentro de un cambio funcional.

Ejemplos:

```
feat: [ID-014] :sparkles: etiquetas OBB para observaciones
fix: :bug: rng faltante en define_observations_limits
```

## Reglas de interacción

- **No commitear ni pushear sin confirmación explícita**, aunque el cambio ya esté verificado.
- Issues y PRs creados o editados por el asistente llevan la label **`ai-review`**
  (nota: esa label todavía **no existe** en el repo; hay que crearla antes del primer uso).
- Sin disclaimers ni notas de autoría automática en el texto de issues/PRs.
- **Nunca nombres de personas** en el cuerpo de issues/PRs.
- Trabajo sobre un Issue → siempre vía **PR**, titulado `<feat|fix>: [ID-XXX] <Título del Issue>`.
  Las ramas de issue siguen el patrón que genera GitHub: `<n>-<slug-del-issue>`.

## Qué NO entra en un commit

Rutas y artefactos locales que se filtran fácil desde los scripts de ejemplo:

- `DESTINY` y demás rutas absolutas de dataset en `generator_save_images_example.py`
  (`/mnt/data3/sponte/...`, `/Users/.../Datasets/...`). Cambiarlas para probar es normal;
  commitearlas no.
- Parámetros del script de ejemplo bajados para una corrida de prueba
  (`BATCHT_CANT = 3`, bloques de kwargs comentados).
- Imágenes de debug generadas por el pipeline.

`.gitignore` ya cubre `.DS_Store`, `*.code-workspace` y `observation_debug.png`. Si aparece
otro artefacto local recurrente, va al `.gitignore` antes que al commit.

Antes de proponer un commit: `git status` y revisar el diff completo, no solo los archivos
que se tocaron a propósito.
