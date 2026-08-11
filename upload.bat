@echo off
title Publication Hugging Face - LTX 2.5 FP8

echo =========================================================================
echo  🤗 Publication du modèle LTX 2.5 FP8 sur Hugging Face Hub
echo =========================================================================
echo.

set "VENV_PYTHON=C:\Users\guill\Documents\ComfyUI\.venv\Scripts\python.exe"
set "SCRIPT_PATH=%~dp0upload_to_huggingface.py"

if not exist "%VENV_PYTHON%" (
    echo [ERREUR] L'environnement virtuel ComfyUI est introuvable.
    pause
    exit /b 1
)

"%VENV_PYTHON%" "%SCRIPT_PATH%" %*

echo.
echo =========================================================================
echo  Appuie sur une touche pour fermer la fenetre...
echo =========================================================================
pause > nul
