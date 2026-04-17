# photos.py — Pipeline de procesamiento de fotos

Convierte HEIC/JPEG a PNG, recorta centrado en la cara detectada y reemplaza el fondo con un color sólido.

## Requisitos

- Python 3.10+
- `yolov11l-face.pt` en la raíz del proyecto (solo para `face-crop` y `pipeline`)

### Instalación de dependencias

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dependencias en `requirements.txt`:**

```
Pillow>=10.0.0
pillow-heif>=0.16.0
torch
ultralytics
rembg
```

> `torch` y `ultralytics` solo se importan con `face-crop` o `pipeline`.  
> `rembg` solo se importa con `bg-change` o `pipeline`.

### Descargar el modelo de detección de caras

```bash
curl -L https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11l-face.pt \
     -o yolov11l-face.pt
```

---

## Uso

```
python photos.py <comando> --input <directorio> [opciones]
```

El directorio de salida se genera automáticamente:
```
<nombre_directorio_entrada>_YYYYMMDD_HHMMSS/
```

---

## Comandos

### `convert` — Convertir HEIC/JPEG a PNG

Convierte todos los archivos `.heic`, `.heif`, `.jpeg` y `.jpg` del directorio de entrada a PNG.

```bash
python photos.py convert --input ./FOTOS
```

**Salida:** `./FOTOS_20260413_153000/`

---

### `face-crop` — Recorte centrado en cara

Detecta la cara con YOLOv11 y genera un recorte cuadrado 1:1 centrado en ella.

```bash
python photos.py face-crop --input ./FOTOS_png
```

**Opciones:**

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--padding` | `2.5` | Multiplicador del alto de cara para el lado del cuadrado |
| `--size` | `1200` | Tamaño final de la imagen en píxeles |
| `--fallback` | `center` | Qué hacer si no se detecta cara: `center` (crop central) o `skip` |

**Salida:** `./FOTOS_png_20260413_153000/`

---

### `bg-change` — Cambiar el fondo

Elimina el fondo de cada imagen con `rembg` y lo reemplaza con un color sólido.

```bash
python photos.py bg-change --input ./FOTOS_cropped --bg-color "#FFFFFF"
```

**Opciones:**

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--bg-color` | `#ECECEC` | Color de fondo en hex |

**Salida:** `./FOTOS_cropped_20260413_153000/`

---

### `pipeline` — Flujo completo

Ejecuta `convert → face-crop → bg-change` en secuencia. La remoción de fondo es implícita — solo hay que pasar `--bg-color`.

```bash
python photos.py pipeline --input ./FOTOS --bg-color "#FFFFFF"
```

**Opciones:**

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--bg-color` | `#ECECEC` | Color de fondo final |
| `--padding` | `2.5` | Multiplicador del alto de cara para el cuadrado |
| `--size` | `1200` | Tamaño final en píxeles |
| `--fallback` | `center` | Qué hacer si no se detecta cara: `center` o `skip` |

**Estructura de salida:**

```
FOTOS_20260413_153000/
├── _converted/        ← PNGs de la conversión (paso 1)
├── _cropped/          ← Recortes cuadrados (paso 2)
├── foto1.png          ← Resultado final con fondo cambiado (paso 3)
├── foto2.png
└── ...
```

---

## Ejemplos rápidos

```bash
# Solo convertir
python photos.py convert --input ./FOTOS

# Recortar caras con padding ajustado, ignorar fotos sin cara
python photos.py face-crop --input ./FOTOS_out --padding 3.0 --fallback skip

# Cambiar fondo a blanco
python photos.py bg-change --input ./FOTOS_cropped --bg-color "#FFFFFF"

# Pipeline completo con fondo gris claro
python photos.py pipeline --input ./FOTOS --bg-color "#ECECEC"

# Pipeline con fondo blanco, output a 800px, ignorar sin cara
python photos.py pipeline --input ./FOTOS --bg-color "#FFFFFF" --size 800 --fallback skip
```

---

## Notas

- El modelo YOLOv11 corre en **MPS** (Apple Silicon) si está disponible, de lo contrario en CPU.
- `rembg` usa el modelo `isnet-general-use` y se descarga automáticamente la primera vez.
- Los directorios `_converted/` y `_cropped/` dentro del output del pipeline son intermedios y se pueden eliminar después de verificar el resultado final.
