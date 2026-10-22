"""Подготовка данных: выгрузка Jigsaw, фильтры, страты по разметке, выборка целевых текстов и пула соседей.

Тексты Jigsaw нельзя перераспространять (правила соревнования Kaggle), поэтому в репозитории хранятся только
скрипты и идентификаторы (`data/processed/*.parquet` в .gitignore). Выборка детерминирована (DATA_SEED).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

COMPETITION = "jigsaw-unintended-bias-in-toxicity-classification"
TRAIN_COLS = ["id", "comment_text", "target", "toxicity_annotator_count", "severe_toxicity", "obscene",
              "identity_attack", "insult", "threat", "sexual_explicit", "article_id"]
DTYPES = {"id": "int32", "target": "float32", "toxicity_annotator_count": "int16", "article_id": "int32",
          "severe_toxicity": "float32", "obscene": "float32", "identity_attack": "float32",
          "insult": "float32", "threat": "float32", "sexual_explicit": "float32"}
KEEP = ["id", "text", "target", "label", "stratum", "toxicity_annotator_count", "n_words", "n_tokens",
        "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit", "article_id"]


def download_raw(raw_dir: Path = C.DATA_RAW) -> Path:
    """Скачивает файлы соревнования (нужен токен KAGGLE_API_TOKEN в .env и принятые правила на сайте Kaggle)."""
    if (raw_dir / "train.csv").exists():
        return raw_dir
    import kagglehub
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env", override=True)
    assert os.getenv("KAGGLE_API_TOKEN"), "Нет KAGGLE_API_TOKEN в .env"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return Path(kagglehub.competition_download(COMPETITION, output_dir=str(raw_dir)))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\r", " ").replace("\n", " ")).strip()


def stratum(t: float):
    """clear_toxic: target ≥ 0.8; clear_nontoxic: target = 0; ambiguous: 0.4 ≤ target ≤ 0.6; иначе не используется."""
    if t >= 0.8:
        return "clear_toxic"
    if t == 0:
        return "clear_nontoxic"
    if 0.4 <= t <= 0.6:
        return "ambiguous"
    return None


def load_filtered(raw_dir: Path) -> pd.DataFrame:
    """Читает train.csv частями (память!), оставляет тексты с достаточным числом разметчиков, применяет фильтры."""
    parts = []
    for chunk in pd.read_csv(raw_dir / "train.csv", usecols=TRAIN_COLS, dtype=DTYPES, chunksize=200_000):
        parts.append(chunk[chunk["toxicity_annotator_count"] >= C.MIN_ANNOTATORS])
    df = pd.concat(parts, ignore_index=True)
    del parts

    ind_path = raw_dir / "toxicity_individual_annotations.csv"
    if ind_path.exists():          # индивидуальная разметка: доля «токсично» среди разметчиков (для проверки спорности)
        keep_ids, parts = set(df["id"]), []
        for chunk in pd.read_csv(ind_path, usecols=["id", "worker", "toxic"],
                                 dtype={"id": "int32", "worker": "int32", "toxic": "int8"}, chunksize=2_000_000):
            parts.append(chunk[chunk["id"].isin(keep_ids)])
        ind = pd.concat(parts, ignore_index=True)
        agg = (ind.assign(is_toxic=(ind["toxic"] > 0).astype(int)).groupby("id")
                  .agg(n_workers=("worker", "nunique"), share_toxic=("is_toxic", "mean")).reset_index())
        df = df.merge(agg, on="id", how="left")

    df["text"] = df["comment_text"].map(normalize)
    df["n_words"] = df["text"].str.split().str.len()
    df["non_ascii"] = df["text"].map(lambda t: sum(ord(ch) > 127 for ch in t) / max(len(t), 1))
    df["uniq_words"] = df["text"].map(lambda t: len(set(t.lower().split())) / max(len(t.split()), 1))
    mask = ((df["toxicity_annotator_count"] >= C.MIN_ANNOTATORS)
            & df["n_words"].between(C.MIN_WORDS, C.MAX_WORDS)
            & ~df["text"].str.contains(r"https?://|www\.", case=False, regex=True)
            & (df["text"].str.len() > 0)
            & (df["non_ascii"] <= C.MAX_NON_ASCII_SHARE)
            & (df["uniq_words"] >= C.MIN_UNIQ_WORD_SHARE))
    clean = df.loc[mask].drop_duplicates(subset="text").copy()

    import tiktoken
    enc = tiktoken.get_encoding(C.TOKENIZER)
    clean["n_tokens"] = [len(enc.encode(t)) for t in clean["text"]]
    clean = clean[clean["n_tokens"] / clean["n_words"] <= C.MAX_TOKENS_PER_WORD].copy()
    clean["stratum"] = clean["target"].map(stratum)
    clean["label"] = (clean["target"] >= 0.5).astype(int)
    return clean.dropna(subset=["stratum"])


def _take(frame: pd.DataFrame, n: int, seed: int = C.DATA_SEED) -> pd.DataFrame:
    if len(frame) < n:
        raise ValueError(f"Не хватает текстов: нужно {n}, есть {len(frame)}")
    return frame.sample(n=n, random_state=seed)


def sample_dataset(clean: pd.DataFrame):
    """300 целевых (75 токсичных + 75 нетоксичных явных + 150 спорных) и пул из 800 соседей (400 + 400)."""
    tox, non, amb = (clean[clean["stratum"] == s] for s in ("clear_toxic", "clear_nontoxic", "ambiguous"))
    clear = pd.concat([_take(tox, C.N_CLEAR_TARGET_PER_CLASS), _take(non, C.N_CLEAR_TARGET_PER_CLASS)])
    targets = pd.concat([clear.assign(target_group="clear"), _take(amb, C.N_AMBIG_TARGET).assign(target_group="ambiguous")])
    used = set(targets["id"])
    neighbors = pd.concat([_take(tox[~tox["id"].isin(used)], C.N_NEIGH_PER_CLASS),
                           _take(non[~non["id"].isin(used)], C.N_NEIGH_PER_CLASS)])
    assert not set(neighbors["id"]) & used
    extra = [c for c in ("n_workers", "share_toxic") if c in clean.columns]
    return targets[KEEP + extra + ["target_group"]], neighbors[KEEP + extra]


def build_and_save(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = download_raw(raw_dir or C.DATA_RAW)
    targets, neighbors = sample_dataset(load_filtered(Path(raw_dir)))
    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(C.DATA_PROC / "targets.parquet", index=False)
    neighbors.to_parquet(C.DATA_PROC / "neighbors_pool.parquet", index=False)
    return targets, neighbors


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(C.DATA_PROC / "targets.parquet"), pd.read_parquet(C.DATA_PROC / "neighbors_pool.parquet"))


def text_lookup(targets: pd.DataFrame, neighbors: pd.DataFrame) -> pd.Series:
    return pd.concat([targets, neighbors]).set_index("id")["text"]
