"""Планы батчей: какие тексты, в каком порядке и с какой долей токсичных соседей уходят в каждый запрос.

Общие свойства всех планов
* батч = 1 целевой + 8 соседей; соседи подобраны по длине (в токенах) к целевому: допуск ±15 %,
  при нехватке кандидатов расширяется до ±20 % и ±30 % (допуск записан в колонке `length_tol`);
* сосед не из той же статьи (`article_id`), что целевой;
* наборы соседей вложены: батч с k токсичными = первые k токсичных + первые (8−k) нетоксичных из одного списка.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config as C

K = C.K_NEIGHBORS


def _matched(pool: pd.DataFrame, target: pd.Series, tol: float) -> pd.DataFrame:
    lo, hi = target["n_tokens"] * (1 - tol), target["n_tokens"] * (1 + tol)
    return pool[pool["n_tokens"].between(lo, hi) & (pool["article_id"] != target["article_id"])]


def _candidates(pools: dict[str, pd.DataFrame], t: pd.Series):
    """Для целевого возвращает ({имя_пула: кандидаты}, допуск) с минимальным допуском, при котором везде ≥ 8 кандидатов."""
    for tol in C.LENGTH_TOLERANCES:
        cands = {name: _matched(p, t, tol) for name, p in pools.items()}
        if all(len(c) >= K for c in cands.values()):
            return cands, tol
    return None, None


def _warn_skipped(skipped):
    if skipped:
        warnings.warn(f"Пропущено целевых без соседей подходящей длины даже при ±{C.LENGTH_TOLERANCES[-1]:.0%}: "
                      f"{len(skipped)} — {', '.join(skipped)}")


def build_h1_plan(targets: pd.DataFrame, neighbors: pd.DataFrame, n_reps: int = 3, seed: int = C.H1_SEED) -> pd.DataFrame:
    """Основной эксперимент (H1/H4): 5 уровней доли токсичных соседей × 3 повтора с разными наборами соседей.

    Позиция целевого определяется повтором (1, 5, 9): позиция и набор соседей смешаны, поэтому влияние
    позиции отдельно оценивается в эксперименте `h1c` (build_position_plan), где позиция полностью скрещена."""
    rng = np.random.default_rng(seed)
    pools = {"tox": neighbors[neighbors["label"] == 1], "non": neighbors[neighbors["label"] == 0]}
    rows, skipped = [], []
    for _, t in targets.iterrows():
        cands, tol = _candidates(pools, t)
        if cands is None:
            skipped.append(f"{t['id']} ({t['n_tokens']} ток.)")
            continue
        for r in range(n_reps):
            tox_ids = rng.choice(cands["tox"]["id"].to_numpy(), K, replace=False)
            non_ids = rng.choice(cands["non"]["id"].to_numpy(), K, replace=False)
            pos = C.H1_POSITIONS[r % len(C.H1_POSITIONS)]
            for k in C.N_TOXIC_LEVELS:
                neigh = list(tox_ids[:k]) + list(non_ids[:K - k])
                rng.shuffle(neigh)
                items = neigh[:pos - 1] + [t["id"]] + neigh[pos - 1:]
                rows.append(dict(batch_id=f"h1_{t['id']}_k{k}_r{r}", target_id=t["id"], n_toxic=k, prop_toxic=k / K,
                                 rep=r, target_pos=pos, item_ids=[int(x) for x in items], length_tol=tol))
    _warn_skipped(skipped)
    return pd.DataFrame(rows)


def build_position_plan(targets: pd.DataFrame, neighbors: pd.DataFrame, levels=C.H1C_LEVELS, positions=C.H1C_POSITIONS,
                        n_sets: int = 1, seed: int = C.H1C_SEED_PLAN) -> pd.DataFrame:
    """Эксперимент про позицию (H5, H1c): для каждой пары (целевой, набор соседей, уровень) порядок соседей фиксируется
    ОДИН раз, и целевой вставляется на каждую позицию. Батчи в ячейке различаются только позицией целевого."""
    rng = np.random.default_rng(seed)
    pools = {"tox": neighbors[neighbors["label"] == 1], "non": neighbors[neighbors["label"] == 0]}
    rows, skipped = [], []
    for _, t in targets.iterrows():
        cands, tol = _candidates(pools, t)
        if cands is None:
            skipped.append(f"{t['id']} ({t['n_tokens']} ток.)")
            continue
        for s in range(n_sets):
            tox_ids = rng.choice(cands["tox"]["id"].to_numpy(), K, replace=False)
            non_ids = rng.choice(cands["non"]["id"].to_numpy(), K, replace=False)
            for k in levels:
                neigh = list(tox_ids[:k]) + list(non_ids[:K - k])
                rng.shuffle(neigh)
                for pos in positions:
                    items = neigh[:pos - 1] + [t["id"]] + neigh[pos - 1:]
                    rows.append(dict(batch_id=f"h1b_{t['id']}_k{k}_s{s}_p{pos}", target_id=t["id"], n_toxic=k,
                                     prop_toxic=k / K, rep=s, target_pos=pos, item_ids=[int(x) for x in items],
                                     length_tol=tol))
    _warn_skipped(skipped)
    return pd.DataFrame(rows)


def sample_h1c_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """100 целевых для эксперимента про позицию: 25 явно токсичных, 25 явно нетоксичных, 50 спорных."""
    seed = C.H1C_SEED_TARGETS
    clear_tox = targets[(targets["target_group"] == "clear") & (targets["label"] == 1)]
    clear_non = targets[(targets["target_group"] == "clear") & (targets["label"] == 0)]
    ambig = targets[targets["target_group"] == "ambiguous"]
    return pd.concat([clear_tox.sample(C.H1C_N_CLEAR_PER_CLASS, random_state=seed),
                      clear_non.sample(C.H1C_N_CLEAR_PER_CLASS, random_state=seed),
                      ambig.sample(C.H1C_N_AMBIG, random_state=seed)])


def build_h1c_plan(targets: pd.DataFrame, neighbors: pd.DataFrame) -> pd.DataFrame:
    """5 400 батчей = 300 целевых × 2 уровня (0 и 100 %) × 9 позиций.

    Первые 100 целевых (`sample_h1c_targets`) и их планы совпадают с первым этапом эксперимента (его ответы уже в кэше);
    остальные 200 планируются отдельно (свой seed), поэтому добавление не меняет уже построенные батчи."""
    core = sample_h1c_targets(targets)
    rest = targets[~targets["id"].isin(core["id"])]
    parts = []
    for sub, seed in ((core, C.H1C_SEED_PLAN), (rest, C.H1C_SEED_PLAN_EXT)):
        if len(sub):
            parts.append(build_position_plan(sub, neighbors, levels=C.H1C_LEVELS, positions=C.H1C_POSITIONS,
                                             n_sets=1, seed=seed))
    plan = pd.concat(parts, ignore_index=True)
    order = {t: i for i, t in enumerate(pd.concat([core, rest])["id"])}
    return (plan.assign(_o=plan["target_id"].map(order)).sort_values(["_o", "n_toxic", "target_pos"])
                .drop(columns="_o").reset_index(drop=True))


def repeat_subset(h1_plan: pd.DataFrame) -> pd.DataFrame:
    """300 батчей основного плана (по 60 на каждый уровень) для проверки самосогласованности модели."""
    per = C.REPEAT_SUBSET_SIZE // h1_plan["prop_toxic"].nunique()
    sub = h1_plan.groupby("prop_toxic", group_keys=False).sample(per, random_state=C.REPEAT_SUBSET_SEED)
    return sub.sort_values("batch_id")[["batch_id"]].merge(h1_plan, on="batch_id")


# ---------------------------------------------------------------------------------------------
# H3 (черновик: определения условий и «нейтральных» соседей согласовать с преподавателем)
# ---------------------------------------------------------------------------------------------
H3_COMPOSITIONS = ("toxic0", "neutral", "toxic100")
H3_TARGET_POS = 5


def build_h3_plan(targets: pd.DataFrame, neighbors: pd.DataFrame, neutral: pd.DataFrame,
                  conditions: tuple[str, ...], seed: int = 21) -> pd.DataFrame:
    """Для каждого целевого и состава соседей (0 % токсичных / нейтральные / 100 % токсичных) набор и порядок соседей
    фиксируются ОДИН раз и используются во всех условиях подачи: условия различаются только промптом."""
    rng = np.random.default_rng(seed)
    pools = {"tox": neighbors[neighbors["label"] == 1], "non": neighbors[neighbors["label"] == 0], "neu": neutral}
    rows, skipped = [], []
    for _, t in targets.iterrows():
        cands, tol = _candidates(pools, t)
        if cands is None:
            skipped.append(f"{t['id']} ({t['n_tokens']} ток.)")
            continue
        for comp, pool in zip(H3_COMPOSITIONS, ("non", "neu", "tox")):
            neigh = [int(x) for x in rng.choice(cands[pool]["id"].to_numpy(), K, replace=False)]
            pos = H3_TARGET_POS
            items = neigh[:pos - 1] + [int(t["id"])] + neigh[pos - 1:]
            for cond in conditions:
                rows.append(dict(batch_id=f"h3_{t['id']}_{comp}_{cond}", target_id=int(t["id"]), composition=comp,
                                 condition=cond, n_toxic={"toxic0": 0, "neutral": 0, "toxic100": 8}[comp],
                                 prop_toxic={"toxic0": 0.0, "neutral": 0.0, "toxic100": 1.0}[comp], rep=0,
                                 target_pos=pos, item_ids=items, length_tol=tol))
    _warn_skipped(skipped)
    return pd.DataFrame(rows)
