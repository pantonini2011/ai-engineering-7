"""Grounding contra los fragmentos RECUPERADOS en cada traza (no contra los documentos completos).

Para cada turno: junta el contenido de los documentos de sus spans `chroma.retriever`
(atributo retrieval.documents) y revisa cada número de la respuesta final:
  - en_fragmentos: aparece en lo que Chroma devolvió en ESA traza → respaldado;
  - en_pregunta: lo dio el usuario;
  - sin_respaldo_directo: ni en los fragmentos ni en la pregunta → hay que ver si es un
    cálculo (verificable) o un indicio de alucinación.

Uso: python analisis/grounding_vs_fragmentos.py m7-ejercicio-sesion2
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from phoenix.client import Client

NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def numeros(texto: str) -> set[str]:
    return {n.replace(",", ".") for n in NUMERO.findall(texto)}


def main(proyecto: str) -> None:
    df = Client().spans.get_spans_dataframe(project_identifier=proyecto, limit=20000, timeout=60)
    turnos = json.loads((Path("traces") / "traza_delegacion.json").read_text(encoding="utf-8"))["turnos"]
    raices = df[df.parent_id.isna()].sort_values("start_time")
    salida = []
    for (_, raiz), turno in zip(raices.iterrows(), turnos, strict=True):
        assert raiz["attributes.session.id"] == turno["thread_id"]
        traza = df[(df["context.trace_id"] == raiz["context.trace_id"]) & (df.name == "chroma.retriever")]
        fragmentos = [
            d["document.content"]
            for docs in traza["attributes.retrieval.documents"].dropna()
            for d in docs
            if d.get("document.content")
        ]
        texto_frag = "\n".join(fragmentos)
        en_frag, en_preg = numeros(texto_frag), numeros(turno["usuario"])
        resp = numeros(turno["respuesta_final"])
        fila = {
            "sesion": turno["thread_id"],
            "fragmentos_recuperados": len(fragmentos),
            "en_fragmentos": sorted(resp & en_frag, key=float),
            "en_pregunta": sorted((resp - en_frag) & en_preg, key=float),
            "sin_respaldo_directo": sorted(resp - en_frag - en_preg, key=float),
        }
        salida.append(fila)
        print(f"\n{fila['sesion']} ({fila['fragmentos_recuperados']} fragmentos)")
        print("  en fragmentos:       ", fila["en_fragmentos"])
        print("  en la pregunta:      ", fila["en_pregunta"])
        print("  sin respaldo directo:", fila["sin_respaldo_directo"])
    destino = Path(__file__).parent / proyecto / "grounding_vs_fragmentos.json"
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {destino}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m7-ejercicio-sesion2")
