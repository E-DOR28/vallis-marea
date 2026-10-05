"""Inferencia del router afinado (R2): encoder multilingue con cabeza lineal.

Es fine-tuning de un **encoder** (cientos de millones de parametros) para
clasificar intencion. No es un LLM generativo y no sustituye a la cascada:
reemplaza solo el nivel 1 de `predecir.py`, y solo cuando `VM_ROUTER=afinado`.

El artefacto es un directorio con:

    model.safetensors   pesos del encoder (prefijo `encoder.`) y de la cabeza
                        (prefijo `cabeza.`)
    config.json         configuracion del encoder
    tokenizer*, *.model archivos del tokenizador
    etiquetas.json      orden de los logits, modelo base, umbral y metadatos
    README.md           model card

Se carga sin `trust_remote_code` y solo desde `safetensors`. Si el origen es un
repositorio del Hub se exige una revision fijada (`usuario/repo@<hash>`):
descargar "la ultima version" de un modelo que se ejecuta en produccion no es
reproducible ni auditable.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_PATRONES_DESCARGA = ["*.json", "*.safetensors", "*.model", "*.txt", "README.md"]
_REVISION = re.compile(r"^[0-9a-f]{40}$")

_CLASE: Any = None


def clase_clasificador() -> Any:
    """Define el modulo perezosamente para no importar torch al arrancar."""
    global _CLASE
    if _CLASE is not None:
        return _CLASE
    import torch

    class Clasificador(torch.nn.Module):
        def __init__(self, encoder: Any, n_clases: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.encoder = encoder
            self.drop = torch.nn.Dropout(dropout)
            self.cabeza = torch.nn.Linear(encoder.config.hidden_size, n_clases)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.cabeza(self.drop(self.pooling(input_ids, attention_mask)))

        def pooling(self, input_ids: Any, attention_mask: Any) -> Any:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1)

    _CLASE = Clasificador
    return _CLASE


def probas_de(modelo: Any, tokenizador: Any, textos: Sequence[str], *, prefijo: str,
              max_len: int, lote: int = 64) -> np.ndarray:
    """Softmax del modelo para una lista de textos, en CPU y sin gradientes."""
    import torch

    modelo.eval()
    salida: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            enc = tokenizador([prefijo + t for t in textos[i : i + lote]], padding=True,
                              truncation=True, max_length=max_len, return_tensors="pt")
            logits = modelo(enc["input_ids"], enc["attention_mask"])
            salida.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.vstack(salida)


def embeddings_de(modelo: Any, tokenizador: Any, textos: Sequence[str], *, prefijo: str,
                  max_len: int, lote: int = 64) -> np.ndarray:
    """Promedio de tokens normalizado (el encoder sin afinar, para el baseline B3)."""
    import torch

    modelo.eval()
    salida: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            enc = tokenizador([prefijo + t for t in textos[i : i + lote]], padding=True,
                              truncation=True, max_length=max_len, return_tensors="pt")
            v = modelo.pooling(enc["input_ids"], enc["attention_mask"])
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            salida.append(v.cpu().numpy())
    return np.vstack(salida)


# ---------------------------------------------------------------------------
# Carga del artefacto
# ---------------------------------------------------------------------------
@dataclass
class ModeloAfinado:
    modelo: Any
    tokenizador: Any
    clases: list[str]
    meta: dict[str, Any]

    @property
    def umbral(self) -> float | None:
        u = self.meta.get("umbral_confianza")
        return float(u) if u is not None else None

    def probas(self, textos: Sequence[str]) -> np.ndarray:
        return probas_de(self.modelo, self.tokenizador, list(textos),
                         prefijo=self.meta.get("prefijo", ""),
                         max_len=int(self.meta.get("max_len", 64)))


def resolver_origen(origen: str, token: str | None = None) -> Path:
    """Devuelve un directorio local a partir de una ruta o de `repo@revision`."""
    ruta = Path(origen)
    if ruta.exists():
        return ruta
    repo, _, revision = origen.partition("@")
    if "/" not in repo:
        raise FileNotFoundError(f"No existe la ruta {origen!r} y no parece un repositorio del Hub")
    if not _REVISION.match(revision):
        raise ValueError(
            "Un modelo del Hub debe indicarse como 'usuario/repo@<hash de 40 caracteres>': "
            "la revision fijada es obligatoria."
        )
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, revision=revision, token=token or None,
                                  allow_patterns=_PATRONES_DESCARGA))


def cargar(origen: str, token: str | None = None) -> ModeloAfinado:
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    directorio = resolver_origen(origen, token)
    meta = json.loads((directorio / "etiquetas.json").read_text(encoding="utf-8"))
    clases = list(meta["clases"])

    cfg = AutoConfig.from_pretrained(directorio, trust_remote_code=False)
    encoder = AutoModel.from_config(cfg, trust_remote_code=False)
    modelo = clase_clasificador()(encoder, len(clases))
    estado = load_file(str(directorio / "model.safetensors"))
    modelo.load_state_dict(estado, strict=True)
    modelo.eval()

    tok = AutoTokenizer.from_pretrained(directorio, trust_remote_code=False)
    return ModeloAfinado(modelo=modelo, tokenizador=tok, clases=clases, meta=meta)


_ACTUAL: ModeloAfinado | None = None
_CERROJO = threading.Lock()


def modelo_actual(origen: str, token: str | None = None) -> ModeloAfinado:
    """Carga una sola vez (el orquestador atiende peticiones concurrentes)."""
    global _ACTUAL
    if _ACTUAL is None:
        with _CERROJO:
            if _ACTUAL is None:
                _ACTUAL = cargar(origen, token)
    return _ACTUAL
