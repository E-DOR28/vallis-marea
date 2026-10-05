"""Publica el router afinado (R2) en un repositorio PRIVADO del Hub de Hugging Face.

    python scripts/publicar_router_hf.py                 # simulacion: no sube nada
    python scripts/publicar_router_hf.py --subir         # sube y verifica que sea privado
    python scripts/publicar_router_hf.py --subir --repo usuario/nombre

Necesita `HF_TOKEN` con permiso de escritura en `.env`. El token nunca se imprime.
Al terminar escribe la revision (hash de 40 caracteres) que debe fijarse en
`VM_ROUTER_MODELO=usuario/repo@<hash>`: `core/router/afinado.py` no descarga
"la ultima version" de un modelo que se ejecuta en produccion.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from core import config  # noqa: E402  (carga .env)

CARPETA = RAIZ / "data" / "modelos" / "router_afinado"
TARJETA = RAIZ / "documentacion" / "model_card_router_afinado.md"
ARCHIVOS = ["config.json", "etiquetas.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
NOMBRE_REPO = "vallis-marea-router-e5-small"


def plan() -> list[tuple[str, Path]]:
    """Archivos a subir como (ruta en el repo, ruta local). Falla si falta alguno."""
    ops = [(nombre, CARPETA / nombre) for nombre in ARCHIVOS] + [("README.md", TARJETA)]
    faltan = [str(ruta.relative_to(RAIZ)) for _, ruta in ops if not ruta.is_file()]
    if faltan:
        raise SystemExit("Faltan archivos: " + ", ".join(faltan))
    return ops


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subir", action="store_true", help="sube de verdad; sin esto solo simula")
    ap.add_argument("--repo", help=f"usuario/nombre; por defecto <tu usuario>/{NOMBRE_REPO}")
    args = ap.parse_args()

    ops = plan()
    total = sum(r.stat().st_size for _, r in ops)
    print(f"Se subirian {len(ops)} archivos ({total / 2**20:.0f} MiB) a un repositorio privado:")
    for destino, ruta in ops:
        print(f"  {destino:<24} {ruta.stat().st_size / 2**20:8.1f} MiB")
    print(f"  SHA-256 de model.safetensors: {sha256(CARPETA / 'model.safetensors')}")
    print(f"HF_TOKEN definido en .env: {'si' if config.HF_TOKEN else 'no'}")

    if not args.subir:
        print("\nSimulacion: no se subio nada. Agrega --subir para publicar.")
        return 0
    if not config.HF_TOKEN:
        print("\nFalta HF_TOKEN en .env (token con permiso de escritura).", file=sys.stderr)
        return 2

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=config.HF_TOKEN)
    usuario = api.whoami()["name"]
    repo = args.repo or f"{usuario}/{NOMBRE_REPO}"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    info = api.create_commit(
        repo_id=repo, repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(r)) for d, r in ops],
        commit_message="Router afinado de R2 (resultado negativo, no adoptado)",
    )
    if not api.model_info(repo).private:
        print(f"ATENCION: {repo} NO es privado. Cambialo en la configuracion del repositorio.", file=sys.stderr)
        return 3
    print(f"\nPublicado en https://huggingface.co/{repo} (privado)")
    print(f"Revision: {info.oid}")
    print(f"VM_ROUTER_MODELO={repo}@{info.oid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
