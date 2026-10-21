'''
Vuelca a una carpeta un dataset en formato YOLO (imagenes, etiquetas AABB u OBB y data.yaml)
generado con SpectrumLabeledSequence. La configuracion se pasa por linea de comandos;
`uv run generator_save_images_example.py --help` lista las opciones. Para mas control sobre
el generador, usar gsssp.save_yolo_dataset desde codigo.
'''

import argparse

from gsssp import LabelClass, LabelFormat, ResizeMode, save_yolo_dataset

CLASSES = {c.yolo_name: c for c in LabelClass}


def parse_classes(text):
    names = [n.strip().lower() for n in text.split(",")]
    unknown = [n for n in names if n not in CLASSES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"clases desconocidas: {', '.join(map(repr, unknown))}. Opciones: {', '.join(CLASSES)}."
        )
    return tuple(dict.fromkeys(CLASSES[n] for n in names))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("output_dir", help="carpeta de salida; se crean images/, labels/ y data.yaml")
parser.add_argument("--batch-size", type=int, default=32, help="imagenes por lote (default 32)")
parser.add_argument(
    "--batch-count", type=int, default=3000,
    help="lotes a generar; el total es batch-size * batch-count (default 3000)",
)
parser.add_argument(
    "--begin-num", type=int, default=0,
    help="numero desde el que arranca la numeracion, para ampliar un dataset sin pisarlo (default 0)",
)
parser.add_argument("--seed", type=int, default=0, help="semilla del generador (default 0)")
parser.add_argument(
    "--label-format", choices=[f.name.lower() for f in LabelFormat], default="aabb",
    help="formato de las etiquetas (default aabb)",
)
parser.add_argument(
    "--label-classes", type=parse_classes, default=(LabelClass.OBSERVATION,),
    help=f"clases a etiquetar, separadas por coma: {', '.join(CLASSES)} (default observacion)",
)
parser.add_argument(
    "--resize-mode", choices=[m.name.lower() for m in ResizeMode], default="none",
    help="como se lleva cada placa a 640x640; stretch con obb se rechaza (default none)",
)
args = parser.parse_args()

total = save_yolo_dataset(
    args.output_dir,
    batch_count=args.batch_count,
    batch_size=args.batch_size,
    begin_num=args.begin_num,
    seed=args.seed,
    label_format=LabelFormat[args.label_format.upper()],
    label_classes=args.label_classes,
    resize_mode=ResizeMode[args.resize_mode.upper()],
)
print(f"{total} imagenes en {args.output_dir}")
