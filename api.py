from __future__ import annotations

import logging
import shutil
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from photos import _convert, _bg_change, _face_crop, _circle_crop

import argparse

app = FastAPI()

SUPPORTED_EXTENSIONS = {".heic", ".heif", ".jpeg", ".jpg", ".png"}
TMP = Path("/tmp/photos")
TMP.mkdir(parents=True, exist_ok=True)


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


@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
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

    try:
        for upload in files:
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Archivo no soportado: {upload.filename}",
                )
            dest = input_dir / upload.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)

        args = _fake_args(padding, size, fallback)

        if action == "convert":
            _convert(input_dir, output_dir)

        elif action == "face-crop":
            converted = job_dir / "converted"
            _convert(input_dir, converted)
            _face_crop(converted, output_dir, args)

        elif action == "bg-change":
            converted = job_dir / "converted"
            _convert(input_dir, converted)
            _bg_change(converted, output_dir, bg_color)

        elif action == "circle-crop":
            converted = job_dir / "converted"
            _convert(input_dir, converted)
            _circle_crop(converted, output_dir)

        elif action == "pipeline":
            converted = job_dir / "_converted"
            cropped = job_dir / "_cropped"
            bg = job_dir / "_bg"
            _convert(input_dir, converted)
            _face_crop(converted, cropped, args)
            _bg_change(cropped, bg, bg_color)
            _circle_crop(bg, output_dir)

        else:
            raise HTTPException(status_code=400, detail=f"Acción desconocida: {action}")

        png_files = list(output_dir.glob("*.png"))
        if not png_files:
            raise HTTPException(status_code=422, detail="No se generaron imágenes de salida.")

        zip_path = job_dir / "result.zip"
        _zip_dir(output_dir, zip_path)

        background_tasks.add_task(shutil.rmtree, job_dir, True)
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename="resultado.zip",
            background=background_tasks,
        )

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        logger.error("Error en /process:\n%s", traceback.format_exc())
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory="static", html=True), name="static")
