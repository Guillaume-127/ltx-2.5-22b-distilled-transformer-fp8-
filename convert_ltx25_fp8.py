#!/usr/bin/env python3
"""
Script de conversion FP8 (e4m3fn) optimisé pour le modèle LTX 2.5 (22B Distilled / Base Transformer)
Conçu pour ComfyUI et les modèles DiT (Diffusion Transformer) de Lightricks.

Principales caractéristiques & optimisations :
1. Conversion sélective : Les poids matriciels 2D+ passent en FP8 (float8_e4m3fn), 
   tandis que les biais (1D), couches de normalisation (LayerNorm/RMSNorm), 
   embeddings (pos/time/text) et modulations (AdaLN scale/shift/gate) restent en BF16.
2. Gestion mémoire RAM stricte : Utilisation de state_dict.pop() pour libérer la RAM
   instantanément après chaque conversion, évitant les surcharges mémoire sur 64 Go RAM.
3. Conservation des métadonnées : Copie de l'en-tête metadata d'origine vers le fichier cible.
4. Mode Dry-Run & CLI : Permet d'analyser la répartition des tenseurs sans charger 44 Go en mémoire.
"""

import os
import sys
import time
import argparse
import gc
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Mots-clés des tenseurs sensibles devant RESTER en haute précision (BF16 / FP16)
SENSITIVE_KEYWORDS = [
    "norm",        # LayerNorm, RMSNorm, GroupNorm
    "bias",        # Biases
    "embed",       # pos_embed, time_embed, text_embed, patch_embed, vector_embed, x_embedder
    "scale",       # Scaling vectors / tables
    "shift",       # Shift vectors / tables
    "gate",        # Gating factors (AdaLN gate, MSA gate, MLP gate)
    "modulation",  # AdaLN modulation blocks
    "adaln",       # AdaLN single / modulation weights
    "time_in",     # Video/Audio conditioning input blocks
    "vector_in",   # Vector condition inputs
    "guidance_in", # Guidance conditioning inputs
    "audio_in",    # Dual-stream LTX 2.5 audio input block
    "video_in",    # Dual-stream LTX 2.5 video input block
]

def is_sensitive_tensor(key: str, shape: torch.Size) -> bool:
    """
    Vérifie si un tenseur doit rester en BF16 pour préserver la qualité visuelle.
    """
    # 1) Tous les tenseurs 0D ou 1D (vecteurs de biais, de norm, d'échelle 1D)
    if len(shape) < 2:
        return True
    
    # 2) Vérification par mots-clés sensibles
    key_lower = key.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in key_lower:
            return True
            
    return False

def convert_ltx_to_fp8(
    input_file: str,
    output_file: str,
    fp8_type: str = "e4m3fn",
    dry_run: bool = False
):
    if input_file.endswith(".crdownload"):
        print(f"❌ Erreur : Le fichier '{input_file}' est un téléchargement en cours (.crdownload).")
        print("💡 Attends la fin complète du téléchargement Chrome avant de lancer le script.")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"❌ Erreur : Le fichier d'entrée '{input_file}' est introuvable.")
        print("💡 Vérifie que le fichier .safetensors est bien présent dans ce dossier.")
        sys.exit(1)

    input_size_gb = os.path.getsize(input_file) / (1024 ** 3)
    print("=" * 75)
    print(f"🎬 Conversion FP8 LTX 2.5 pour ComfyUI")
    print(f"📂 Fichier source : {input_file} ({input_size_gb:.2f} GB)")
    print(f"🎯 Fichier cible  : {output_file}")
    print(f"⚡ Format FP8     : torch.float8_{fp8_type}")
    print("=" * 75)

    # Sélection du format PyTorch FP8
    if fp8_type == "e4m3fn":
        target_fp8_dtype = torch.float8_e4m3fn
    elif fp8_type == "e5m2":
        target_fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Format FP8 inconnu : {fp8_type}")

    # 1. Lecture des métadonnées d'origine sans charger les tenseurs
    print("\n🔍 Analyse de l'en-tête du fichier safetensors...")
    metadata = {}
    with safe_open(input_file, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        keys = f.keys()
        
        # Inspection rapide si dry-run
        if dry_run:
            print("\n🧪 Mode DRY-RUN (Analyse sans conversion ni chargement lourd) :")
            converted_cnt = 0
            preserved_cnt = 0
            for k in keys:
                slice_obj = f.get_slice(k)
                shape = slice_obj.get_shape()
                dtype_str = slice_obj.get_dtype()
                
                if dtype_str in ["F32", "F16", "BF16"] and not is_sensitive_tensor(k, shape):
                    converted_cnt += 1
                else:
                    preserved_cnt += 1
                    
            print(f"\n📊 Résultat simulé sur {len(keys)} tenseurs :")
            print(f"   - Tenseurs à convertir en FP8 ({fp8_type}) : {converted_cnt}")
            print(f"   - Tenseurs à conserver en BF16 (Norm/Bias/Embed/AdaLN) : {preserved_cnt}")
            print("=" * 75)
            return

    print(f"ℹ️ {len(metadata)} entrée(s) de métadonnées conservée(s).")
    start_time = time.time()

    # 2. Chargement des poids en mémoire
    print("\n📦 Chargement du state dict BF16 en mémoire RAM...")
    print("   (La RAM va temporairement augmenter jusqu'à ~45 Go avant nettoyage)")
    state_dict = load_file(input_file)
    
    fp8_state_dict = {}
    converted_count = 0
    preserved_count = 0

    print("\n⚡ Conversion en FP8 (avec libération RAM en temps réel)...")
    
    # Stratégie pop() : on extrait chaque tenseur un à un pour libérer le dictionnaire initial
    keys_list = list(state_dict.keys())
    for key in tqdm(keys_list, desc="Traitement des tenseurs", unit="tenseur"):
        tensor = state_dict.pop(key)
        
        if tensor.is_floating_point():
            if not is_sensitive_tensor(key, tensor.shape):
                # Conversion des matrices 2D+ en FP8 e4m3fn
                fp8_state_dict[key] = tensor.to(target_fp8_dtype)
                converted_count += 1
            else:
                # Préservation en BF16 / FP16 d'origine
                fp8_state_dict[key] = tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
                preserved_count += 1
        else:
            fp8_state_dict[key] = tensor

    # Force le Garbage Collector Python à purger la mémoire système inutilisée
    del state_dict
    gc.collect()

    print(f"\n✅ Conversion terminée en {time.time() - start_time:.2f} s")
    print(f"   - Tenseurs convertis en FP8 ({fp8_type}) : {converted_count}")
    print(f"   - Tenseurs préservés en BF16 (Norm/Bias/Embed/Modulation) : {preserved_count}")
    
    # 3. Sauvegarde du fichier Safetensors
    print(f"\n💾 Sauvegarde du nouveau fichier safetensors : {output_file}")
    save_file(fp8_state_dict, output_file, metadata=metadata)
    
    # Libération finale
    del fp8_state_dict
    gc.collect()

    output_size_gb = os.path.getsize(output_file) / (1024 ** 3)
    gain_pct = (1 - (output_size_gb / input_size_gb)) * 100

    print("\n" + "=" * 75)
    print(f"🎉 Modèle FP8 prêt pour ComfyUI !")
    print(f"📏 Taille originale : {input_size_gb:.2f} GB")
    print(f"📉 Taille convertie : {output_size_gb:.2f} GB (Gain de {gain_pct:.1f}%)")
    print(f"📍 Chemin du fichier : {os.path.abspath(output_file)}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convertit le modèle LTX 2.5 (22B) du format BF16 au FP8 pour ComfyUI."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        help="Chemin du fichier source BF16 .safetensors (défaut: ltx-2.5-22b-distilled-transformer-bf16.safetensors)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="ltx-2.5-22b-distilled-transformer-fp8_e4m3fn.safetensors",
        help="Chemin du fichier de sortie FP8 .safetensors (défaut: ltx-2.5-22b-distilled-transformer-fp8_e4m3fn.safetensors)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["e4m3fn", "e5m2"],
        default="e4m3fn",
        help="Sous-type FP8 (défaut: e4m3fn, recommandé pour RTX 4090)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Effectue une analyse des tenseurs sans charger les 44 Go ni écrire de fichier."
    )

    args = parser.parse_args()
    convert_ltx_to_fp8(args.input, args.output, fp8_type=args.format, dry_run=args.dry_run)
