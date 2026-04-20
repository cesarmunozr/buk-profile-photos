FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libheif-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download rembg model so it's baked into the image (no cold-start download)
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

COPY . .

# Download YOLOv11 face detection model (not committed to git — large binary)
RUN curl -L https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11l-face.pt \
    -o /app/yolov11l-face.pt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
