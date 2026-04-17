#!/usr/bin/env python3
"""
photos.py — Pipeline de procesamiento de fotos de perfil.

Comandos individuales:
  convert      Convierte HEIC/HEIF/JPEG/JPG a PNG.
  face-crop    Recorta 1:1 centrado en la cara detectada.
  bg-change    Elimina el fondo y lo reemplaza con un color sólido.
  circle-crop  Aplica máscara circular (RGBA, esquinas transparentes).
  pipeline     Ejecuta convert → face-crop → bg-change → circle-crop.

El directorio de salida se genera automáticamente:
  <directorio_entrada>_YYYYMMDD_HHMMSS/

Ejemplos:
  # 1. Solo convertir HEIC/JPG a PNG
  python photos.py convert --input ./fotos

  # 2. Buscar cara y recortar (requiere PNGs)
  python photos.py face-crop --input ./fotos_20240101_120000
  python photos.py face-crop --input ./fotos_20240101_120000 --padding 3.0 --size 800

  # 3. Borrar fondo y poner color sólido (requiere PNGs)
  python photos.py bg-change --input ./fotos_20240101_120000
  python photos.py bg-change --input ./fotos_20240101_120000 --bg-color "#FFFFFF"

  # 4. Recortar en círculo (requiere PNGs)
  python photos.py circle-crop --input ./fotos_20240101_120000

  # 5. Pipeline completo: convierte → recorta cara → cambia fondo → círculo
  python photos.py pipeline --input ./fotos
  python photos.py pipeline --input ./fotos --bg-color "#FFFFFF" --size 800
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw
import pillow_heif


BASE = Path(__file__).parent
MODEL_PATH = BASE / "yolov11l-face.pt"
SUPPORTED_CONVERT = {".heic", ".heif", ".jpeg", ".jpg"}
CENTER_Y_OFFSET = 0.10  # Desplazamiento vertical del crop hacia arriba (fracción del alto de cara)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_output_dir(input_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = input_dir.parent / f"{input_dir.name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


# ── Convert ────────────────────────────────────────────────────────────────────

def _convert(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED_CONVERT]

    if not files:
        print("  No se encontraron archivos compatibles (.heic, .heif, .jpeg, .jpg).")
        return

    ok = failed = 0
    for src in sorted(files):
        dst = output_dir / (src.stem + ".png")
        try:
            if src.suffix.lower() in {".heic", ".heif"}:
                heif_file = pillow_heif.read_heif(src)
                image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
            else:
                image = Image.open(src)
            image.save(dst, format="PNG")
            print(f"  OK   {src.name} → {dst.name}")
            ok += 1
        except Exception as e:
            print(f"  ERR  {src.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n  Convertidos: {ok} | Errores: {failed}")


def cmd_convert(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).resolve()
    output_dir = make_output_dir(input_dir)

    print(f"\nConvertiendo: {input_dir}")
    print(f"Salida:       {output_dir}\n")
    _convert(input_dir, output_dir)
    print(f"\nSalida: {output_dir}")


# ── Face Crop ──────────────────────────────────────────────────────────────────

def _load_face_model():
    import torch
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_PATH}\n"
            "Descárgalo con:\n"
            "  curl -L https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11l-face.pt"
            f" -o {MODEL_PATH}"
        )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Cargando YOLOv11-face en {device.upper()}...")
    model = YOLO(str(MODEL_PATH))
    model.to(device)
    return model, device


def _square_crop(img: Image.Image, cx: float, cy: float, side: float, fill: tuple) -> Image.Image:
    w, h = img.size
    half = side / 2
    src_x1, src_y1 = cx - half, cy - half
    src_x2, src_y2 = cx + half, cy + half

    if src_x1 >= 0 and src_y1 >= 0 and src_x2 <= w and src_y2 <= h:
        return img.crop((int(src_x1), int(src_y1), int(src_x2), int(src_y2)))

    canvas = Image.new("RGB", (int(side), int(side)), fill)
    clip_x1, clip_y1 = max(0, src_x1), max(0, src_y1)
    clip_x2, clip_y2 = min(w, src_x2), min(h, src_y2)
    if clip_x2 > clip_x1 and clip_y2 > clip_y1:
        patch = img.crop((int(clip_x1), int(clip_y1), int(clip_x2), int(clip_y2)))
        canvas.paste(patch, (int(clip_x1 - src_x1), int(clip_y1 - src_y1)))
    return canvas


def _center_crop_fallback(img: Image.Image, fill: tuple) -> Image.Image:
    w, h = img.size
    return _square_crop(img, w / 2, h / 2, min(w, h), fill=fill)


def _face_crop(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.png"))
    if not files:
        print(f"  No se encontraron imágenes PNG en {input_dir}")
        return

    face_model, device = _load_face_model()

    print(f"\n  Procesando {len(files)} imágenes → {output_dir}\n")

    ok = no_face = failed = 0
    fill = (0, 0, 0)

    for src in files:
        dst = output_dir / src.name
        try:
            img = Image.open(src).convert("RGB")
            results = face_model(str(src), device=device, verbose=False)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                if args.fallback == "skip":
                    print(f"  SKIP  {src.name}  (sin cara)")
                    no_face += 1
                    continue
                cropped = _center_crop_fallback(img, fill=fill)
                print(f"  WARN  {src.name}  (sin cara → crop central)")
                no_face += 1
            else:
                best_idx = int(boxes.conf.argmax())
                x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
                face_w, face_h = x2 - x1, y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2 - face_h * CENTER_Y_OFFSET
                side = max(face_h * args.padding, face_w, face_h)
                cropped = _square_crop(img, cx, cy, side, fill=fill)
                conf = float(boxes.conf[best_idx])
                print(f"  OK    {src.name}  (conf={conf:.2f}, bbox={int(face_w)}x{int(face_h)})")
                ok += 1

            cropped = cropped.resize((args.size, args.size), Image.LANCZOS)
            cropped.save(dst, format="PNG", optimize=True)

        except Exception as e:
            print(f"  ERR   {src.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n  Listo: {ok} con cara | {no_face} sin cara | {failed} errores")


def cmd_face_crop(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).resolve()
    output_dir = make_output_dir(input_dir)

    print(f"\nRecortando por cara: {input_dir}")
    print(f"Salida:              {output_dir}\n")
    _face_crop(input_dir, output_dir, args)
    print(f"\nSalida: {output_dir}")


# ── BG Change ──────────────────────────────────────────────────────────────────

def _apply_circle_mask(img: Image.Image) -> Image.Image:
    """Aplica una máscara circular a una imagen cuadrada. Retorna RGBA con esquinas transparentes."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, w - 1, h - 1), fill=255)
    result = img.convert("RGBA")
    result.putalpha(mask)
    return result


def _bg_change(input_dir: Path, output_dir: Path, bg_color_hex: str) -> None:
    from rembg import new_session, remove

    output_dir.mkdir(parents=True, exist_ok=True)
    bg_color = hex_to_rgb(bg_color_hex)

    files = sorted(input_dir.glob("*.png"))
    if not files:
        print(f"  No se encontraron imágenes PNG en {input_dir}")
        return

    print(f"  Cargando rembg (isnet-general-use)...")
    session = new_session("isnet-general-use")
    print(f"  Fondo: #{bg_color_hex.lstrip('#').upper()}  RGB{bg_color}")
    print(f"\n  Procesando {len(files)} imágenes → {output_dir}\n")

    ok = failed = 0
    for src in files:
        dst = output_dir / src.name
        try:
            img = Image.open(src).convert("RGB")
            rgba = remove(img, session=session)
            background = Image.new("RGB", rgba.size, bg_color)
            background.paste(rgba, mask=rgba.split()[3])
            background.save(dst, format="PNG", optimize=True)
            print(f"  OK   {src.name}")
            ok += 1
        except Exception as e:
            print(f"  ERR  {src.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n  Listo: {ok} procesadas | {failed} errores")


def _circle_crop(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.png"))
    if not files:
        print(f"  No se encontraron imágenes PNG en {input_dir}")
        return

    print(f"  Procesando {len(files)} imágenes → {output_dir}\n")

    ok = failed = 0
    for src in files:
        dst = output_dir / src.name
        try:
            img = Image.open(src).convert("RGB")
            result = _apply_circle_mask(img)
            result.save(dst, format="PNG", optimize=True)
            print(f"  OK   {src.name}")
            ok += 1
        except Exception as e:
            print(f"  ERR  {src.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n  Listo: {ok} procesadas | {failed} errores")


def cmd_circle_crop(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).resolve()
    output_dir = make_output_dir(input_dir)

    print(f"\nRecorte circular: {input_dir}")
    print(f"Salida:           {output_dir}\n")
    _circle_crop(input_dir, output_dir)
    print(f"\nSalida: {output_dir}")


def cmd_bg_change(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).resolve()
    output_dir = make_output_dir(input_dir)

    print(f"\nCambiando fondo: {input_dir}")
    print(f"Salida:          {output_dir}\n")
    _bg_change(input_dir, output_dir, args.bg_color)
    print(f"\nSalida: {output_dir}")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def cmd_pipeline(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).resolve()
    output_dir = make_output_dir(input_dir)
    converted_dir = output_dir / "_converted"
    cropped_dir = output_dir / "_cropped"
    bg_dir = output_dir / "_bg"

    print(f"\nPipeline completo: {input_dir}")
    print(f"Salida:            {output_dir}\n")

    print("=" * 60)
    print(" Paso 1/4 — Conversión a PNG")
    print("=" * 60)
    _convert(input_dir, converted_dir)

    print()
    print("=" * 60)
    print(" Paso 2/4 — Recorte por cara")
    print("=" * 60)
    _face_crop(converted_dir, cropped_dir, args)

    print()
    print("=" * 60)
    print(" Paso 3/4 — Cambio de fondo")
    print("=" * 60)
    _bg_change(cropped_dir, bg_dir, args.bg_color)

    print()
    print("=" * 60)
    print(" Paso 4/4 — Recorte circular")
    print("=" * 60)
    _circle_crop(bg_dir, output_dir)

    print(f"\nSalida final: {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _add_crop_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--padding",
        type=float,
        default=2.5,
        help="Multiplicador del alto de cara para el lado del cuadrado (default: 2.5)",
    )
    p.add_argument(
        "--size",
        type=int,
        default=1200,
        help="Tamaño final en píxeles (default: 1200)",
    )
    p.add_argument(
        "--fallback",
        choices=["center", "skip"],
        default="center",
        help="Qué hacer si no se detecta cara: center | skip (default: center)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── convert ──
    p_conv = sub.add_parser("convert", help="Convierte HEIC/HEIF/JPEG/JPG a PNG")
    p_conv.add_argument("--input", required=True, metavar="DIR", help="Directorio de entrada")
    p_conv.set_defaults(func=cmd_convert)

    # ── face-crop ──
    p_face = sub.add_parser("face-crop", help="Recorta 1:1 centrado en cara detectada")
    p_face.add_argument("--input", required=True, metavar="DIR", help="Directorio de entrada (PNG)")
    _add_crop_args(p_face)
    p_face.set_defaults(func=cmd_face_crop)

    # ── circle-crop ──
    p_circle = sub.add_parser("circle-crop", help="Recorta en forma circular (RGBA, esquinas transparentes)")
    p_circle.add_argument("--input", required=True, metavar="DIR", help="Directorio de entrada (PNG)")
    p_circle.set_defaults(func=cmd_circle_crop)

    # ── bg-change ──
    p_bg = sub.add_parser("bg-change", help="Elimina el fondo y lo reemplaza con un color sólido")
    p_bg.add_argument("--input", required=True, metavar="DIR", help="Directorio de entrada (PNG)")
    p_bg.add_argument(
        "--bg-color",
        type=str,
        default="#ECECEC",
        help="Color de fondo en hex (default: #ECECEC)",
    )
    p_bg.set_defaults(func=cmd_bg_change)

    # ── pipeline ──
    p_pipe = sub.add_parser(
        "pipeline",
        help="Pipeline completo: convert → face-crop → bg-change → circle-crop",
    )
    p_pipe.add_argument("--input", required=True, metavar="DIR", help="Directorio de entrada")
    p_pipe.add_argument(
        "--bg-color",
        type=str,
        default="#ECECEC",
        help="Color de fondo en hex (default: #ECECEC)",
    )
    _add_crop_args(p_pipe)
    p_pipe.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
