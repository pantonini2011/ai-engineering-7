"""Sesión 1 (descartada por jerarquía rota): empareja cada span HTTP del SDK de Anthropic
(`anthropic.messages.create`) con su span `ChatAnthropic` de LangChain para medir el
overhead local del wrapper sobre la llamada de red."""

from pathlib import Path

import pandas as pd
from phoenix.client import Client

df = Client().spans.get_spans_dataframe(project_identifier="m7-ejercicio-sesion1", limit=10000)
df["ms"] = (df.end_time - df.start_time).dt.total_seconds() * 1000
P, C = "attributes.llm.token_count.prompt", "attributes.llm.token_count.completion"
sdk, lc = df[df.name == "anthropic.messages.create"], df[df.name == "ChatAnthropic"]

pares = []
for _, s in sdk.iterrows():
    cand = lc[(lc[P] == s[P]) & (lc[C] == s[C])]
    if len(cand):
        c = cand.iloc[(cand.start_time - s.start_time).abs().argmin()]
        pares.append({"http_ms": s.ms, "langchain_ms": c.ms, "overhead_ms": c.ms - s.ms, "prompt": s[P], "completion": s[C]})

p = pd.DataFrame(pares)
print(f"spans SDK={len(sdk)} ChatAnthropic={len(lc)} emparejados={len(p)}")
print(p.describe().round(1).to_string())
p.to_csv(Path(__file__).with_suffix(".csv"), index=False)
