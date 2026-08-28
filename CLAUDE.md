# GASP — Generator for Astronomical Spectroscopic Plates

Generador de imágenes sintéticas de escaneos de placas espectroscópicas, con etiquetas
listas para entrenar modelos de detección (YOLO / Keras-CV). Se usa como **librería
importable** y también vía scripts de ejemplo.

> La carpeta local se llama `GSSSP` y el paquete Python sigue siendo `gsssp`, pero el repo
> en GitHub ya es **`Sansanto2000/GASP`** (pasó por `G3SP` en el medio). Si el `git remote`
> todavía apunta a `GSSSP.git`, funciona por redirección pero conviene actualizarlo:
> `git remote set-url origin https://github.com/Sansanto2000/GASP.git`.
> No renombrar el paquete de forma oportunista: es un cambio propio, no algo que se cuela
> en otro commit.

## Estructura

```
src/gsssp/
  __init__.py                        API pública del paquete (__all__)
  drawing.py                         draw_observation: dibuja una observación
  spectra.py                         spectral_function, Fading, planck_like
  noise.py                           add_realistic_noise, add_plate_edge, Position
  labels.py                          formateadores yolov11, edges_of_labels_relxywh
  geometry.py                        cajas envolventes y camino OBB (ObservationLimit)
  debug.py                           visualize_observations
  observationArtist.py               shim de compatibilidad: solo reexporta
  generators/
    __init__.py
    spectrumLabeledSequence.py       keras.utils.Sequence que produce lotes (imagen, etiquetas)
src/test.py                          script manual de debug de define_observations_limits
generator_save_images_example.py     ejemplo: volcar un dataset a disco
assets/                              imágenes del README
```

Los imports son siempre **`gsssp.*`** (paquete instalado en modo editable), nunca
`src.gsssp.*`: esa segunda forma dependía del cwd y podía cargar el módulo dos veces.

`observationArtist.py` era el archivo único de ~800 líneas; hoy es solo un shim que
reexporta desde los módulos nuevos. Se puede borrar cuando no queden imports apuntándole.

### Camino OBB (WIP)

`geometry.py` tiene el trabajo a medio hacer hacia etiquetas OBB (4 esquinas) y componentes
por clase (`observacion` / `science` / `lamp`). No es deuda accidental:

- `define_observations_limits()` calcula las esquinas de cada observación y el generador ya
  lo llama, pero **el resultado todavía no se usa aguas abajo**: las observaciones se siguen
  posicionando con el código viejo. Conectarlo es el próximo paso.
- `define_observation_components_limits()` y `ObservationLimit.define_components_limits()`
  son `pass`.
- **Antes de conectarlo hay que decidir la convención de signo del ángulo**: `cv2.boxPoints`
  y `cv2.getRotationMatrix2D` rotan al revés, y hoy el AABB tapa el error porque es simétrico.

### Convenciones de código

- Docstrings y comentarios en **español** (sin tildes en buena parte del código existente;
  seguir el estilo del archivo que se edita).
- Los módulos de `src/gsssp/` usan indentación de 4 espacios; `spectrumLabeledSequence.py`
  usa **2**. Respetar la del archivo, no unificar de paso.
- Nombres de funciones en **snake_case**, sin excepciones. Los nombres viejos en camelCase
  (`drawObservation`, `labelDictToYolov11Format`, `labelListToYolov11Format`) y
  `rotate_point` fueron **eliminados**, no quedan alias. Los **parámetros** de
  `draw_observation` siguen en camelCase (`distanceBetweenParts`, `baseGrey`): renombrarlos
  es un cambio propio y aparte.
- Aleatoriedad: **no se usan `random` ni `np.random` globales en ningún lado**. Toda función
  que sortee algo recibe un `np.random.Generator` como parámetro *keyword-only* `rng`, con
  default `None` que resuelve a `np.random.default_rng()`. `SpectrumLabeledSequence` deriva
  el suyo de `(seed, idx)`, así que el lote es función pura del índice: reproducible, y sin
  estado compartido entre hilos ni procesos. Al agregar una función que sortee, seguir el
  mismo patrón; volver a los globales rompe la reproducibilidad y el paralelismo.
- Parámetros del generador: todos *keyword-only*, con rangos como tuplas `(min, max)`.
  Al agregar uno, sumarlo también al docstring de `__init__` y a `self.<nombre>`.

## Entorno

- Python **3.10** (`requires-python = ">=3.10,<3.11"`), gestionado con **uv**.
- TensorFlow 2.16.1 con dependencia condicional por plataforma: `tensorflow-macos` +
  `tensorflow-metal` en darwin, `tensorflow[and-cuda]` en linux. **No tocar ese bloque
  de `pyproject.toml` sin considerar ambas plataformas** — se desarrolla en macOS y se
  entrena en Linux con GPU.
- `uv.lock` está versionado: si cambian dependencias, el lock va **en el mismo commit**.

### Configuración de las corridas

Los parámetros que cambian por corrida viven en un `.env` en la raíz, no en el código:
`GASP_OUTPUT_DIR` (obligatoria, sin ella el script corta), `GASP_BATCH_SIZE`,
`GASP_BATCH_COUNT` y `GASP_BEGIN_NUM`. `.env.example` está commiteado y hace de
documentación; `.env` está en `.gitignore`.

```bash
cp .env.example .env      # y completar GASP_OUTPUT_DIR
uv run generator_save_images_example.py
```

Lo que ya esté exportado en el entorno **pisa** al `.env`, así que se puede cambiar un valor
puntual sin editar el archivo.

**El entorno lo leen los scripts de entrada, nunca la librería.** `src/gsssp/` no accede a
`os.environ` en ningún lado y debe seguir así: recibe todo por parámetro y el script traduce
las variables a esos parámetros. Es el mismo criterio que el `rng` explícito — la dependencia
entra por argumento, no por estado global.

## Formatos de etiquetas

| Formato | Anatomía | Notas |
|---|---|---|
| YOLO (AABB) | `<class> <x> <y> <w> <h>` | class entero; resto normalizado [0-1], centro + tamaño. Lo que produce hoy el pipeline. |
| OBB | `<class> <x1> <y1> ... <x4> <y4>` | 4 esquinas normalizadas. Objetivo del camino WIP. |

Clases previstas: `observacion`, `science`, `lamp`.

---

# Git

## Reglas de interacción

- **No commitear ni pushear sin confirmación explícita**, aunque el cambio ya esté verificado.
  El flujo es: hacer el cambio, verificarlo, mostrarlo, y **esperar la aprobación** antes de
  commitear.

## Flujo de trabajo sobre un Issue

Vale para cualquier Issue, siempre igual:

1. **Rama propia**, nombrada `ID-XXX` con el número del Issue. Se trabaja ahí, nunca sobre
   `main` ni `dev`.
2. **Vincular la rama al Issue** en GitHub, para que aparezca en la sección *Development*.
3. Commits con el formato de abajo, cada uno esperando aprobación.
4. Cuando están todos los commits, **crear el PR**. El merge lo ejecuta una persona,
   contra `dev` y **con rebase**.

## Mensajes de commit

```
<feat|fix>: [ID-XXX] descripción breve
```

- **Solo `feat` o `fix`**, nunca los dos.
- `[ID-XXX]` con el número del Issue, que normalmente coincide con el nombre de la rama.
  Si no hay Issue, se omite por completo.
- La descripción es **breve** y puede llevar un emoji, en formato shortcode
  (`:sparkles:`, `:bug:`, `:memo:`, `:wrench:`, `:fire:`, `:zap:`, `:recycle:`, `:boom:`),
  consistente con el historial.
- Un commit = **un solo propósito**, y chico. Nada de arrastrar reformateos, renombres
  de proyecto o ajustes de parámetros de prueba dentro de un cambio funcional.

Ejemplos:

```
feat: [ID-014] :sparkles: etiquetas OBB para observaciones
fix: [ID-008] :bug: rng explicito en add_plate_edge
fix: :fire: eliminar rotate_point, funcion sin uso
```

El PR se titula `<feat|fix>: [ID-XXX] <Título del Issue>`.

## Issues

Aplican tanto a este repo como a **midusi/PlateUNLP**, de donde salió parte de este código.

### Buscar antes de crear

Antes de abrir un issue nuevo, buscar si ya existe:

```bash
gh search issues --repo Sansanto2000/GASP "términos"
```

Los astrónomos del equipo reportan **síntomas** desde el lado observacional antes de que se
encuentre la causa en el código, así que es común que el síntoma ya esté cargado. Si ya
existe, comentar ahí el diagnóstico en vez de duplicar.

### Respaldar con números, no solo con lectura de código

Si el caso se puede reproducir corriendo el código (`uv run <script>`, o
`.venv/bin/python` para un script suelto), hacerlo y citar los valores medidos. Leer el
código alcanza para una hipótesis; correrlo la convierte en confirmación.

### Estructura del cuerpo

- Historia de usuario: *"Como [rol], quiero [necesidad], para [motivo]."*
- `## Problema` — qué pasa hoy, con cita de código o valores medidos si corresponde.
- `## Posible origen` — solo si hay una causa raíz identificada. No inventarla si no se tiene.
- `## Posible solución` — **UNA** recomendación concreta, no una lista neutra de alternativas.
  Si hay un camino obviamente más simple, decirlo explícitamente. Ser concreto sobre el
  **mecanismo** ("acotando el alto según el ángulo"), no solo sobre el resultado
  ("que las etiquetas entren en el canvas").
- `## Relacionado` / `## Depende de` — referencias cruzadas (`#XXX`). Al agregar una,
  comentar también en el issue del otro lado para que el link se vea desde ambos.

### Tono

Profesional y formal. **Nunca el nombre de una persona** en el cuerpo ni en un comentario:
ni como responsable, ni como fuente de la idea, ni como quien lo reportó ("como señaló X"
queda afuera). La asignación se maneja con el campo *assignee* o el milestone de GitHub.

Sin disclaimers ni notas de autoría automática en el texto.

### Etiquetas y milestone

- **`ai-review`** en todo lo que el asistente cree o edite (issue nuevo, comentario, cuerpo
  editado): es la señal de "pendiente de revisión humana", y reemplaza a cualquier marcador
  tipo 🤖 dentro del texto. **Todavía no existe en este repo**, hay que crearla antes del
  primer uso.
- **`usable-minima`**: marca lo necesario para una primera versión usable. **No
  autoasignársela** — es una decisión de priorización, no algo que se infiera del bug.
- Milestone **"Tesis V. Dome"**: agrupa un plan de trabajo completo, incluido el trabajo de
  soporte, no solo lo de una persona. Asignarlo cuando el issue sea parte de ese plan.
  (Ni esta label ni este milestone existen hoy en este repo; sí en PlateUNLP.)

### Gotcha de herramienta

`gh api -f campo=@archivo` **no lee el archivo**: escribe el string literal `@archivo`. El
`@` mágico solo funciona con `-F`/`--field` (mayúscula). Verificar el resultado después de
cualquier `gh api -X PATCH` que lea de un archivo, porque no falla si sale mal.

## Qué NO entra en un commit

Rutas y artefactos locales que se filtran fácil:

- Valores de corrida en `.env`. Ya está en `.gitignore`, así que el riesgo no es commitearlo
  sino **mover un valor de vuelta al código** para probar algo. Si hace falta un parámetro
  nuevo, va al `.env` y al `.env.example`, no hardcodeado en el script.
- Bloques de kwargs comentados en los scripts de ejemplo, dejados de una prueba.
- Imágenes de debug generadas por el pipeline.

`.gitignore` ya cubre `.DS_Store`, `*.code-workspace`, `observation_debug.png`, `.env` y
`.envrc`. Si aparece otro artefacto local recurrente, va al `.gitignore` antes que al commit.

Antes de proponer un commit: `git status` y revisar el diff completo, no solo los archivos
que se tocaron a propósito.
