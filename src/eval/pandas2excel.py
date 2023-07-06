"""MLW-OCR Projekt.

Pandas2Excel Script.
"""

import argparse

import pandas as pd
from torchmetrics import CharErrorRate

metric = CharErrorRate()
parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="Transform json to xlsx")
parser.add_argument(
    "--src-path",
    required=True,
    help="Path to source file.",
)
parser.add_argument("--trg-path", required=True, help="Path to target file.")
parser.add_argument("--mdl-name", required=True, help="Model name.")

args: argparse.Namespace = parser.parse_args()
src_path: str = args.src_path
trg_path: str = args.trg_path
mdl_name: str = args.mdl_name

df = pd.read_json(src_path)
df["CER"] = list(
    map(lambda e: metric(preds=e[1][2], target=e[1][1]).item(), df.iterrows())
)
df["Correct"] = list(map(lambda e: 1 if e[1][2] == e[1][1] else 0, df.iterrows()))
df = df.rename(columns={"lemma": "Label", "prediction": "Prediction"})
df["Model"] = [mdl_name] * len(df)
df = df.drop(columns=["id"])
df.to_excel(trg_path, index=False)
