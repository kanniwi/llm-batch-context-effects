import warnings

import numpy as np
import pandas as pd

K_NEIGHBORS = 8                      # соседей в батче (+1 целевой = 9 текстов)
N_TOXIC_LEVELS = [0, 2, 4, 6, 8]     # = доли 0 / 25 / 50 / 75 / 100 %
TARGET_POSITIONS = [1, 5, 9]         # позиция целевого в повторе r = 0, 1, 2 (начало / середина / конец)
LENGTH_TOLERANCES = [0.15, 0.20, 0.30]  # ±15 %


def _matched(pool, target, tol):
    lo, hi = target["n_tokens"] * (1 - tol), target["n_tokens"] * (1 + tol)
    return pool[(pool["n_tokens"].between(lo, hi)) & (pool["article_id"] != target["article_id"])]


def build_h1_plan(targets: pd.DataFrame, neighbors: pd.DataFrame, n_reps: int = 3, seed: int = 42) -> pd.DataFrame:
    """для каждого целевого и каждого повтора r берём 8 токсичных и 8 нетоксичных соседей близкой длины.
    батч с k токсичными = первые k токсичных + первые (8-k) нетоксичных (вложенные наборы).
    целевой стоит на позиции TARGET_POSITIONS[r], соседи в случайном порядке.
    целевые, для которых нет 8+8 соседей даже при максимальном допуске, пропускаются (с предупреждением)."""
    rng = np.random.default_rng(seed)
    tox_pool = neighbors[neighbors["label"] == 1]
    non_pool = neighbors[neighbors["label"] == 0]
    rows, skipped = [], []
    for _, t in targets.iterrows():
        for tol in LENGTH_TOLERANCES:
            tox_c, non_c = _matched(tox_pool, t, tol), _matched(non_pool, t, tol)
            if len(tox_c) >= K_NEIGHBORS and len(non_c) >= K_NEIGHBORS:
                break
        else:
            skipped.append(f"{t['id']} ({t['n_tokens']} ток.)")
            continue
        for r in range(n_reps):
            tox_ids = rng.choice(tox_c["id"].to_numpy(), K_NEIGHBORS, replace=False)
            non_ids = rng.choice(non_c["id"].to_numpy(), K_NEIGHBORS, replace=False)
            pos = TARGET_POSITIONS[r % len(TARGET_POSITIONS)]
            for k in N_TOXIC_LEVELS:
                neigh = list(tox_ids[:k]) + list(non_ids[:K_NEIGHBORS - k])
                rng.shuffle(neigh)
                items = neigh[:pos - 1] + [t["id"]] + neigh[pos - 1:]
                rows.append(dict(batch_id=f"h1_{t['id']}_k{k}_r{r}", target_id=t["id"], n_toxic=k,
                                 prop_toxic=k / K_NEIGHBORS, rep=r, target_pos=pos,
                                 item_ids=[int(x) for x in items], length_tol=tol))
    if skipped:
        warnings.warn(f"Пропущено целевых без соседей подходящей длины даже при ±{LENGTH_TOLERANCES[-1]:.0%}: "
                      f"{len(skipped)} — {', '.join(skipped)}")
    return pd.DataFrame(rows)


def build_position_plan(targets: pd.DataFrame, neighbors: pd.DataFrame, levels=(0, 4, 8), positions=(1, 5, 9),
                        n_sets: int = 1, seed: int = 7) -> pd.DataFrame:
    """Эксперимент H1b: позиция целевого ПОЛНОСТЬЮ скрещена с долей токсичных соседей и не связана с повтором.
    Для каждого целевого и каждого набора соседей (s) и уровня k берём ОДИН и тот же упорядоченный список из 8 соседей
    и вставляем целевой на каждую из позиций. Значит, внутри одной ячейки (цель, набор, уровень) батчи отличаются
    ТОЛЬКО позицией целевого: те же тексты в том же порядке."""
    rng = np.random.default_rng(seed)
    tox_pool = neighbors[neighbors["label"] == 1]
    non_pool = neighbors[neighbors["label"] == 0]
    rows, skipped = [], []
    for _, t in targets.iterrows():
        for tol in LENGTH_TOLERANCES:
            tox_c, non_c = _matched(tox_pool, t, tol), _matched(non_pool, t, tol)
            if len(tox_c) >= K_NEIGHBORS and len(non_c) >= K_NEIGHBORS:
                break
        else:
            skipped.append(f"{t['id']} ({t['n_tokens']} ток.)")
            continue
        for s in range(n_sets):
            tox_ids = rng.choice(tox_c["id"].to_numpy(), K_NEIGHBORS, replace=False)
            non_ids = rng.choice(non_c["id"].to_numpy(), K_NEIGHBORS, replace=False)
            for k in levels:
                neigh = list(tox_ids[:k]) + list(non_ids[:K_NEIGHBORS - k])
                rng.shuffle(neigh)                      # порядок соседей фиксируем один раз для всех позиций
                for pos in positions:
                    items = neigh[:pos - 1] + [t["id"]] + neigh[pos - 1:]
                    rows.append(dict(batch_id=f"h1b_{t['id']}_k{k}_s{s}_p{pos}", target_id=t["id"], n_toxic=k,
                                     prop_toxic=k / K_NEIGHBORS, rep=s, target_pos=pos,
                                     item_ids=[int(x) for x in items], length_tol=tol))
    if skipped:
        warnings.warn(f"Пропущено целевых без соседей подходящей длины: {len(skipped)} — {', '.join(skipped)}")
    return pd.DataFrame(rows)
