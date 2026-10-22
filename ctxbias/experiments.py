"""Описание экспериментов: что именно запрашивается у модели.

Каждый эксперимент порождает список «заданий» (jobs): одиночные запросы (комментарий без соседей) и батчи
(целевой + 8 соседей). Задание однозначно определяет запрос к API; ответ на него кэшируется.

| эксперимент   | гипотезы        | что делает                                                                         |
|---------------|-----------------|------------------------------------------------------------------------------------|
| h1            | H1, H4, H2      | 300 целевых × 5 уровней × 3 повтора = 4 500 батчей + 1 100 одиночных               |
| h1c           | H5 (+H1)        | 300 целевых × 2 уровня × 9 позиций = 5 400 батчей (позиция скрещена с составом)     |
| repeats       | шум модели      | 300 батчей основного набора и 300 одиночных, sample = 0, 1, 2 при рабочей T         |
| uncertainty   | H2 (спорность)  | 300 одиночных × 10 ответов при T = 1: доля TOXIC = неуверенность модели             |
| h3            | H3 (черновик)   | 5 условий подачи × 3 состава соседей × 300 целевых                                 |
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from . import config as C
from . import data, plans
from .prompts import H3_CONDITIONS, SYSTEM, build_h3_prompt, build_user_prompt


@dataclass
class Context:
    """Данные и планы, общие для всех экспериментов (строятся детерминированно)."""
    targets: pd.DataFrame
    neighbors: pd.DataFrame
    neutral: pd.DataFrame | None = None
    lang: str = "en"

    @classmethod
    def load(cls, lang: str = "en") -> "Context":
        t, n = data.load_processed()
        p = C.DATA_PROC / "neutral_pool.parquet"
        return cls(t, n, pd.read_parquet(p) if p.exists() else None, lang)

    @cached_property
    def text(self) -> pd.Series:
        return data.text_lookup(self.targets, self.neighbors if self.neutral is None else pd.concat([self.neighbors, self.neutral]))

    @cached_property
    def gold(self) -> pd.Series:
        return pd.concat([self.targets, self.neighbors] + ([self.neutral] if self.neutral is not None else [])).set_index("id")["label"]

    @cached_property
    def system(self) -> str:
        return SYSTEM[self.lang]

    @cached_property
    def h1_plan(self) -> pd.DataFrame:
        return plans.build_h1_plan(self.targets, self.neighbors)

    @cached_property
    def h1c_plan(self) -> pd.DataFrame:
        return plans.build_h1c_plan(self.targets, self.neighbors)

    @cached_property
    def repeat_plan(self) -> pd.DataFrame:
        return plans.repeat_subset(self.h1_plan)


def _single_jobs(ctx: Context, ids, samples, temperature, exp):
    return [dict(exp=exp, kind="single", id=str(i), target_id=int(i), system=ctx.system,
                 user=build_user_prompt([ctx.text[i]]), max_tokens=C.MAX_TOKENS_SINGLE, temperature=temperature,
                 sample=s, target_pos=1, n_items=1, item_ids=[int(i)])
            for s in samples for i in ids]


def _batch_jobs(ctx: Context, plan, samples, temperature, exp):
    cols = [c for c in ("prop_toxic", "n_toxic", "rep", "length_tol", "composition", "condition") if c in plan.columns]
    jobs = []
    for s in samples:
        for r in plan.itertuples():
            ids = list(r.item_ids)
            texts = [ctx.text[i] for i in ids]
            if getattr(r, "condition", None) is not None and "condition" in plan.columns:
                system, user = build_h3_prompt(r.condition, texts, [int(ctx.gold[i]) for i in ids], int(r.target_pos))
                mt = C.MAX_TOKENS_BATCH if r.condition == "standard" else 32
            else:
                system, user, mt = ctx.system, build_user_prompt(texts), C.MAX_TOKENS_BATCH
            j = dict(exp=exp, kind="batch", id=r.batch_id, target_id=int(r.target_id), system=system, user=user,
                     max_tokens=mt, temperature=temperature, sample=s, target_pos=int(r.target_pos),
                     n_items=len(ids), item_ids=[int(i) for i in ids])
            j.update({c: getattr(r, c) for c in cols})
            jobs.append(j)
    return jobs


@dataclass(frozen=True)
class Experiment:
    name: str
    hypotheses: str
    default_samples: tuple[int, ...]
    description: str = ""

    def jobs(self, ctx: Context, samples: tuple[int, ...] | None = None) -> list[dict]:
        samples = tuple(self.default_samples if samples is None else samples)
        n = self.name
        T = C.T_WORK
        tids = list(ctx.targets["id"])
        if n == "h1":
            all_ids = tids + list(ctx.neighbors["id"])
            return _single_jobs(ctx, all_ids, samples, T, n) + _batch_jobs(ctx, ctx.h1_plan, samples, T, n)
        if n == "h1c":
            sub = list(ctx.h1c_plan["target_id"].drop_duplicates())
            return _single_jobs(ctx, sub, samples, T, n) + _batch_jobs(ctx, ctx.h1c_plan, samples, T, n)
        if n == "repeats":
            return _single_jobs(ctx, tids, samples, T, n) + _batch_jobs(ctx, ctx.repeat_plan, samples, T, n)
        if n == "uncertainty":
            return _single_jobs(ctx, tids, samples, C.T_UNCERTAINTY, n)
        if n == "h3":
            if ctx.neutral is None:
                raise RuntimeError("Нет data/processed/neutral_pool.parquet: выполните `python -m ctxbias data`")
            plan = plans.build_h3_plan(ctx.targets, ctx.neighbors, ctx.neutral, H3_CONDITIONS)
            return _batch_jobs(ctx, plan, samples, T, n)
        raise KeyError(n)


EXPERIMENTS: dict[str, Experiment] = {e.name: e for e in [
    Experiment("h1", "H1, H4, H2", (C.SAMPLE_BASE,), "Основной: доля токсичных соседей 0–100 %"),
    Experiment("h1c", "H5, H1", (C.SAMPLE_BASE,), "Позиция целевого, скрещённая с составом соседей"),
    Experiment("repeats", "шум модели", (C.SAMPLE_BASE,) + C.SAMPLES_REPEAT, "Самосогласованность при рабочей T"),
    Experiment("uncertainty", "H2", C.SAMPLES_UNCERTAINTY, "10 ответов при T=1: неуверенность модели"),
    Experiment("h3", "H3", (C.SAMPLE_BASE,), "Условия подачи × состав соседей (черновик)"),
]}
