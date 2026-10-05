"""Diagnostico de la curva de validacion (R2, ronda 1). Ver el .json hermano."""
import sys, json, numpy as np
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from core.router import afinar as A, protocolo as P

textos, y, grupos = A._cargar_dataset()
part = A._particiones(y, grupos)[0]
tr, va = P.dividir_validacion(part["entrenamiento"], y, grupos, P.semilla_validacion(part["semilla"], part["particion"]))
cand = A.CANDIDATOS["e5_small"]
tok = AutoTokenizer.from_pretrained(cand["repo"], revision=cand["revision"])
for lr, lote, lr_cab in [(5e-5, 16, 1e-3), (1e-4, 8, 3e-3)]:
    A.HIPER.update(lote=lote, lr_cabeza=lr_cab, paciencia=999, epocas_max=40)
    _, info = A.entrenar_una(cand, tok, textos, y, tr, va, lr, 7, epocas_max=40)
    print(f"lr={lr} lote={lote} lr_cabeza={lr_cab}")
    print("  f1_val por epoca:", info["f1_val_por_epoca"])
