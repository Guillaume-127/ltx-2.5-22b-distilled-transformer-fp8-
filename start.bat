@echo off
title Conversion LTX 2.5 BF16 vers FP8 (e4m3fn) pour ComfyUI

echo =========================================================================
echo  Lanceur de Conversion FP8 pour LTX 2.5 (22B Transformer)
echo =========================================================================
echo.

set "VENV_PYTHON=C:\Users\guill\Documents\ComfyUI\.venv\Scripts\python.exe"
set "SCRIPT_PATH=%~dp0convert_ltx25_fp8.py"
set "INPUT_FILE=%~dp0ltx-2.5-22b-distilled-transformer-bf16.safetensors"
set "OUTPUT_FILE=%~dp0ltx-2.5-22b-distilled-transformer-fp8_e4m3fn.safetensors"

if not exist "%VENV_PYTHON%" (
    echo [ERREUR] L'environnement virtuel ComfyUI est introuvable :
    echo %VENV_PYTHON%
    echo.
    pause
    exit /b 1
)

if exist "%INPUT_FILE%.crdownload" (
    echo [ATTENTION] Le telechargement du fichier est toujours EN COURS !
    echo Fichier temporaire trouve : ltx-2.5-22b-distilled-transformer-bf16.safetensors.crdownload
    echo.
    echo Merci d'attendre la fin complete du telechargement Chrome avant de convertir.
    echo.
    pause
    exit /b 1
)

"%VENV_PYTHON%" "%SCRIPT_PATH%" --input "%INPUT_FILE%" --output "%OUTPUT_FILE%" --format e4m3fn

echo.
echo =========================================================================
echo  Conversion terminee ! Appuie sur une touche pour fermer la fenetre...
echo =========================================================================
pause > nul
