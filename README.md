# photos.py — Pipeline de procesamiento de fotos de perfil

Convierte HEIC/JPEG a PNG, recorta centrado en la cara detectada, reemplaza el fondo con un color sólido y recorta en círculo.

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
python photos.py face-crop --input ./FOTOS_png --padding 3.0 --size 800 --fallback skip
```

**Opciones:**

| Opción       | Default  | Descripción                                                       |
| ------------ | -------- | ----------------------------------------------------------------- |
| `--padding`  | `2.5`    | Multiplicador del alto de cara para el lado del cuadrado          |
| `--size`     | `1200`   | Tamaño final de la imagen en píxeles                              |
| `--fallback` | `center` | Qué hacer si no se detecta cara: `center` (crop central) o `skip` |

**Salida:** `./FOTOS_png_20260413_153000/`

---

### `bg-change` — Cambiar el fondo

Elimina el fondo de cada imagen con `rembg` y lo reemplaza con un color sólido.

```bash
python photos.py bg-change --input ./FOTOS_cropped
python photos.py bg-change --input ./FOTOS_cropped --bg-color "#FFFFFF"
```

**Opciones:**

| Opción       | Default   | Descripción           |
| ------------ | --------- | --------------------- |
| `--bg-color` | `#ECECEC` | Color de fondo en hex |

**Salida:** `./FOTOS_cropped_20260413_153000/`

---

### `circle-crop` — Recortar en círculo

Aplica una máscara circular a cada imagen. El resultado es RGBA con esquinas transparentes.

```bash
python photos.py circle-crop --input ./FOTOS_bg
```

**Salida:** `./FOTOS_bg_20260413_153000/`

---

### `pipeline` — Flujo completo

Ejecuta los 4 pasos en secuencia: `convert → face-crop → bg-change → circle-crop`.

```bash
python photos.py pipeline --input ./FOTOS
python photos.py pipeline --input ./FOTOS --bg-color "#FFFFFF" --size 800 --fallback skip
```

**Opciones:**

| Opción       | Default   | Descripción                                        |
| ------------ | --------- | -------------------------------------------------- |
| `--bg-color` | `#ECECEC` | Color de fondo                                     |
| `--padding`  | `2.5`     | Multiplicador del alto de cara para el cuadrado    |
| `--size`     | `1200`    | Tamaño final en píxeles                            |
| `--fallback` | `center`  | Qué hacer si no se detecta cara: `center` o `skip` |

**Estructura de salida:**

```
FOTOS_20260413_153000/
├── _converted/        ← PNGs convertidos (paso 1)
├── _cropped/          ← Recortes cuadrados por cara (paso 2)
├── _bg/               ← Imágenes con fondo cambiado (paso 3)
├── foto1.png          ← Resultado final circular (paso 4)
├── foto2.png
└── ...
```

---

## Ejemplos rápidos

```bash
# 1. Solo convertir HEIC/JPG a PNG
python photos.py convert --input ./FOTOS

# 2. Detectar cara y recortar cuadrado 1:1
python photos.py face-crop --input ./FOTOS_out
python photos.py face-crop --input ./FOTOS_out --padding 3.0 --fallback skip

# 3. Borrar fondo y poner color sólido
python photos.py bg-change --input ./FOTOS_cropped
python photos.py bg-change --input ./FOTOS_cropped --bg-color "#FFFFFF"

# 4. Recortar en círculo (RGBA, esquinas transparentes)
python photos.py circle-crop --input ./FOTOS_bg

# 5. Pipeline completo (los 4 pasos en secuencia)
python photos.py pipeline --input ./FOTOS
python photos.py pipeline --input ./FOTOS --bg-color "#FFFFFF" --size 800 --padding 3.0
```

---

## Referencia de flags

| Flag                   | Comando(s)              | Default   | Descripción                                                                                                                    |
| ---------------------- | ----------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `--size 800`           | `face-crop`, `pipeline` | `1200`    | Tamaño final en píxeles (la imagen queda `N x N`). Útil para reducir el peso del archivo.                                      |
| `--padding 3.0`        | `face-crop`, `pipeline` | `2.5`     | Espacio alrededor de la cara. Multiplicador del alto de cara detectada: `2.5` es ajustado, `3.0` incluye más cuello y hombros. |
| `--fallback skip`      | `face-crop`, `pipeline` | `center`  | Qué hacer si no se detecta cara: `center` recorta el centro de la imagen, `skip` descarta la foto.                             |
| `--bg-color "#FFFFFF"` | `bg-change`, `pipeline` | `#ECECEC` | Color de fondo en hex que reemplaza al fondo removido.                                                                         |

---

## Notas

- El modelo YOLOv11 corre en **MPS** (Apple Silicon) si está disponible, de lo contrario en CPU.
- `rembg` usa el modelo `isnet-general-use` y se descarga automáticamente la primera vez.
- Los directorios `_converted/`, `_cropped/` y `_bg/` dentro del output del pipeline son intermedios y se pueden eliminar después de verificar el resultado final.
