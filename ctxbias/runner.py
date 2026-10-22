"""Запуск эксперимента: задания → ответы модели (через кэш) → таблица результатов + манифест запуска."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from . import __version__
from . import config as C
from .experiments import EXPERIMENTS, Context
from .llm import ask_many, daily_limit_hit, is_cached
from .prompts import parse_labels, p_toxic_from_logprobs


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=C.ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("+uncommitted" if dirty.stdout.strip() else "")
    except Exception:
        return None


def count_requests(exp_name: str, model_key: str, ctx: Context, samples=None) -> dict:
    """Сколько запросов в эксперименте и сколько из них уже в кэше (то есть бесплатно)."""
    jobs = EXPERIMENTS[exp_name].jobs(ctx, samples)
    cached = sum(is_cached(model_key, j["system"], j["user"], j["temperature"], j["max_tokens"], j["sample"]) for j in jobs)
    return dict(experiment=exp_name, total=len(jobs), cached=cached, remaining=len(jobs) - cached)


def _tidy(job: dict, resp: dict, model_key: str) -> dict:
    n = job["n_items"]
    labels, ok = parse_labels(resp["text"], n)
    probs = p_toxic_from_logprobs(resp["tokens"], n) if resp.get("tokens") else [None] * n
    pos = job["target_pos"]
    usage = resp.get("usage") or {}
    row = {k: job[k] for k in ("exp", "kind", "id", "target_id", "sample", "temperature", "target_pos", "n_items", "item_ids")}
    for k in ("prop_toxic", "n_toxic", "rep", "length_tol", "composition", "condition"):
        row[k] = job.get(k)
    row.update(model_key=model_key, y=labels[pos - 1], p=probs[pos - 1], labels=labels, parse_ok=ok,
               finish_reason=resp.get("finish_reason"), error=resp.get("error"), in_tokens=usage.get("input"),
               out_tokens=usage.get("output"), served_model=resp.get("served_model"), temperature_applied=resp.get("temperature_applied", True), raw=resp["text"])
    return row


def run(exp_name: str, model_key: str, samples=None, ctx: Context | None = None, save: bool = True) -> pd.DataFrame:
    """Выполняет эксперимент. Возвращает таблицу (по строке на запрос) и сохраняет её в data/results."""
    ctx = ctx or Context.load()
    exp = EXPERIMENTS[exp_name]
    jobs = exp.jobs(ctx, samples)
    cached_before = sum(is_cached(model_key, j["system"], j["user"], j["temperature"], j["max_tokens"], j["sample"]) for j in jobs)
    rows = []
    # группируем по (sample, температура), чтобы прогресс был понятным; внутри — все задания сразу
    groups: dict = {}
    for j in jobs:
        groups.setdefault((j["sample"], j["temperature"], j["kind"]), []).append(j)
    for (s, T, kind), js in groups.items():
        resp = ask_many(model_key, [dict(j, id=f"{j['kind']}|{j['id']}") for j in js],
                        desc=f"{exp_name} {kind} sample={s} T={T}")
        rows += [_tidy(j, resp[f"{j['kind']}|{j['id']}"], model_key) for j in js]
        if daily_limit_hit(model_key):
            break
    df = pd.DataFrame(rows)
    if save and len(df):
        C.DATA_RES.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["raw"]).to_parquet(C.DATA_RES / f"{exp_name}_{model_key}.parquet", index=False)
        spec = C.MODELS[model_key]
        manifest = dict(
            experiment=exp_name, model_key=model_key, provider=spec.provider, model=spec.model, base_url=spec.base_url,
            model_notes=spec.notes, served_models=sorted({m for m in df["served_model"].dropna().unique()}),
            samples=sorted(df["sample"].unique().tolist()), temperatures_requested=sorted(df["temperature"].unique().tolist()),
            temperature_effective=("как запрошено" if bool(df["temperature_applied"].all()) else "значение по умолчанию у провайдера (API не принял запрошенную температуру; значение неизвестно)"),
            n_requests=len(df), n_cached_before=cached_before, n_new=len(df) - cached_before,
            n_errors=int(df["error"].notna().sum()), n_unparsed=int((~df["parse_ok"] & df["error"].isna()).sum()),
            n_length_truncated=int((df["finish_reason"] == "length").sum()),
            temperature_applied=bool(df["temperature_applied"].all()), daily_limit=daily_limit_hit(model_key), finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            package_version=__version__, git_commit=_git_commit(), python=platform.python_version(), platform=sys.platform)
        (C.DATA_RES / f"{exp_name}_{model_key}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def load_results(exp_name: str, model_key: str) -> pd.DataFrame:
    return pd.read_parquet(C.DATA_RES / f"{exp_name}_{model_key}.parquet")
