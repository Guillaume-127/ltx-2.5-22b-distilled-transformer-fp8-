#!/usr/bin/env python3
"""
Script d'upload automatique vers Hugging Face Hub pour le modèle LTX 2.5 FP8.
Gère l'authentification par token, la création du repo et l'upload résiliant.
"""

import os
import sys
import argparse

# Force l'encodage UTF-8 pour la console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Uploader le modèle LTX 2.5 FP8 sur Hugging Face.")
    parser.add_argument("--repo-id", type=str, help="ID du repo Hugging Face (ex: ton_username/ltx-2.5-22b-distilled-transformer-fp8)")
    parser.add_argument("--token", type=str, help="Token d'accès Hugging Face (hf_...) avec permissions d'écriture (Write)")
    args = parser.parse_args()

    # Vérification du package huggingface_hub
    try:
        from huggingface_hub import HfApi, create_repo, login
    except ImportError:
        print("❌ 'huggingface_hub' n'est pas installé dans ton environnement.")
        print("💡 Installation en cours...")
        os.system(f'"{sys.executable}" -m pip install huggingface_hub')
        from huggingface_hub import HfApi, create_repo, login

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\n🔑 Connexion à Hugging Face")
        print("--------------------------------------------------")
        print("Rends-toi sur https://huggingface.co/settings/tokens")
        print("Crée un token avec les permissions 'WRITE'.")
        print("--------------------------------------------------")
        token = input("👉 Colle ton token Hugging Face (hf_...) : ").strip()

    if not token.startswith("hf_"):
        print("❌ Le token saisi semble invalide (un token commence par 'hf_').")
        sys.exit(1)

    login(token=token)
    api = HfApi()

    # Récupérer le nom d'utilisateur connecté
    user_info = api.whoami(token=token)
    username = user_info["name"]
    print(f"\n✅ Connecté avec succès en tant que : {username}")

    default_repo = f"{username}/ltx-2.5-22b-distilled-transformer-fp8"
    repo_id = args.repo_id or default_repo

    print(f"\n📦 Création / Vérification du dépôt sur Hugging Face : {repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    folder_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(folder_dir, "ltx-2.5-22b-distilled-transformer-fp8_e4m3fn.safetensors")
    readme_file = os.path.join(folder_dir, "README.md")
    script_file = os.path.join(folder_dir, "convert_ltx25_fp8.py")

    print("\n🚀 Début du transfert vers Hugging Face Hub...")
    print("ℹ️ L'upload du fichier de ~21.5 Go utilise Git LFS avec reprise automatique.")

    # 1. Upload du README.md
    if os.path.exists(readme_file):
        print("\n📄 Envoi de la fiche modèle (README.md)...")
        api.upload_file(
            path_or_fileobj=readme_file,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

    # 2. Upload du script de conversion
    if os.path.exists(script_file):
        print("🐍 Envoi du script Python de conversion (convert_ltx25_fp8.py)...")
        api.upload_file(
            path_or_fileobj=script_file,
            path_in_repo="convert_ltx25_fp8.py",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

    # 3. Upload du modèle .safetensors (21.5 Go)
    if os.path.exists(model_file):
        size_gb = os.path.getsize(model_file) / (1024 ** 3)
        print(f"📦 Envoi du modèle FP8 ({size_gb:.2f} Go)... Merci de patienter pendant l'upload.")
        api.upload_file(
            path_or_fileobj=model_file,
            path_in_repo="ltx-2.5-22b-distilled-transformer-fp8_e4m3fn.safetensors",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
    else:
        print(f"⚠️ Fichier modèle introuvable : {model_file}")
        print("   Seules la documentation et la fiche ont été téléversées.")

    print("\n" + "=" * 70)
    print("🎉 PUBLICATION REUSSIE SUR HUGGING FACE !")
    print(f"🔗 URL de ton dépôt : https://huggingface.co/{repo_id}")
    print("=" * 70)

if __name__ == "__main__":
    main()
