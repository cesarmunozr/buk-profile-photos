from __future__ import annotations

import asyncio
import logging
import queue
import shutil
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from photos import _bg_change, _circle_crop, _convert, _face_crop

import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=2)

SUPPORTED_EXTENSIONS = {".heic", ".heif", ".jpeg", ".jpg", ".png"}
TMP = Path("/tmp/photos")
TMP.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict] = {}


def _fake_args(padding: float, size: int, fallback: str) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.padding = padding
    ns.size = size
    ns.fallback = fallback
    return ns


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(source_dir.glob("*.png")):
            zf.write(file, file.name)


def _run_job(
    job_id: str,
    job_dir: Path,
    input_dir: Path,
    output_dir: Path,
    action: str,
    args: argparse.Namespace,
    bg_color: str,
) -> None:
    q: queue.Queue = jobs[job_id]["queue"]

    def log(msg: str) -> None:
        q.put(msg)

    try:
        if action == "convert":
            log("Convirtiendo imágenes a PNG...")
            _convert(input_dir, output_dir, on_file=log)
            log("STEP_OK:Conversión completada")

        elif action == "face-crop":
            converted = job_dir / "converted"
            log("Convirtiendo imágenes a PNG...")
            _convert(input_dir, converted, on_file=log)
            log("STEP_OK:Conversión completada")
            log("Detectando y recortando caras...")
            _face_crop(converted, output_dir, args, on_file=log)
            log("STEP_OK:Recorte de caras completado")

        elif action == "bg-change":
            converted = job_dir / "converted"
            log("Convirtiendo imágenes a PNG...")
            _convert(input_dir, converted, on_file=log)
            log("STEP_OK:Conversión completada")
            log("Eliminando fondo y aplicando color sólido...")
            _bg_change(converted, output_dir, bg_color, on_file=log)
            log("STEP_OK:Cambio de fondo completado")

        elif action == "circle-crop":
            converted = job_dir / "converted"
            log("Convirtiendo imágenes a PNG...")
            _convert(input_dir, converted, on_file=log)
            log("STEP_OK:Conversión completada")
            log("Aplicando recorte circular...")
            _circle_crop(converted, output_dir, on_file=log)
            log("STEP_OK:Recorte circular completado")

        elif action == "pipeline":
            converted = job_dir / "_converted"
            cropped = job_dir / "_cropped"
            bg = job_dir / "_bg"
            log("Paso 1/4 — Convirtiendo imágenes a PNG...")
            _convert(input_dir, converted, on_file=log)
            log("STEP_OK:Conversión completada")
            log("Paso 2/4 — Detectando y recortando caras...")
            _face_crop(converted, cropped, args, on_file=log)
            log("STEP_OK:Recorte de caras completado")
            log("Paso 3/4 — Eliminando fondo...")
            _bg_change(cropped, bg, bg_color, on_file=log)
            log("STEP_OK:Cambio de fondo completado")
            log("Paso 4/4 — Aplicando recorte circular...")
            _circle_crop(bg, output_dir, on_file=log)
            log("STEP_OK:Recorte circular completado")

        png_files = list(output_dir.glob("*.png"))
        if not png_files:
            raise RuntimeError("No se generaron imágenes de salida.")

        if len(png_files) == 1:
            jobs[job_id]["result_path"] = str(png_files[0])
            jobs[job_id]["result_name"] = png_files[0].name
            jobs[job_id]["result_mime"] = "image/png"
        else:
            log("Comprimiendo resultados...")
            zip_path = job_dir / "result.zip"
            _zip_dir(output_dir, zip_path)
            jobs[job_id]["result_path"] = str(zip_path)
            jobs[job_id]["result_name"] = "resultado.zip"
            jobs[job_id]["result_mime"] = "application/zip"

        log("DONE")

    except Exception as exc:
        logger.error("Error en job %s:\n%s", job_id, traceback.format_exc())
        jobs[job_id]["error"] = str(exc)
        log(f"ERROR:{exc}")
    finally:
        jobs[job_id]["finished"] = True


@app.post("/process")
async def process_start(
    files: List[UploadFile] = File(...),
    action: str = Form(...),
    bg_color: str = Form("#ECECEC"),
    size: int = Form(1200),
    padding: float = Form(2.5),
    fallback: str = Form("center"),
):
    job_id = uuid.uuid4().hex
    job_dir = TMP / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Archivo no soportado: {upload.filename}")
        dest = input_dir / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

    args = _fake_args(padding, size, fallback)
    jobs[job_id] = {"queue": queue.Queue(), "zip_path": None, "error": None, "finished": False}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_job, job_id, job_dir, input_dir, output_dir, action, args, bg_color)

    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream_progress(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job no encontrado.")

    async def generate():
        q: queue.Queue = jobs[job_id]["queue"]
        while True:
            try:
                msg = q.get_nowait()
                yield f"data: {msg}\n\n"
                if msg == "DONE" or msg.startswith("ERROR:"):
                    break
            except queue.Empty:
                if jobs[job_id]["finished"] and q.empty():
                    break
                await asyncio.sleep(0.2)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/download/{job_id}")
async def download(background_tasks: BackgroundTasks, job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    if job.get("error"):
        raise HTTPException(status_code=500, detail=job["error"])
    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Archivo no disponible.")

    job_dir = Path(result_path).parent
    background_tasks.add_task(shutil.rmtree, str(job_dir), True)
    background_tasks.add_task(jobs.pop, job_id, None)

    return FileResponse(
        path=result_path,
        media_type=job["result_mime"],
        filename=job["result_name"],
        background=background_tasks,
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
