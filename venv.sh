#!/bin/bash

if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "Entorno virtual creado en el directorio 'venv'"

  source venv/bin/activate
  echo "Entorno virtual activado"

  if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
    echo "Dependencias instaladas."
  fi
else
  echo "El entorno virtual ya existe."
  source venv/bin/activate
  echo "Entorno virtual activado"
fi