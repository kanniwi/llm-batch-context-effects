"""Метрики и статистические тесты. Все функции принимают «тидовую» таблицу результатов (см. runner._tidy).

Принципы (см. docs/methods.md):
  * эффект считается ВНУТРИ целевого комментария (разность долей TOXIC при разном составе соседей);
  * доверительные интервалы — бутстреп по целевым (а не по запросам: повторы одного целевого зависимы);
  * вывод делается по тому, содержит ли интервал 0, а не по попаданию оценки в «собственный» интервал;
  * анализы, не заданные заранее, помечаются exploratory=True.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

RNG_SEED = 20240601
N_BOOT = 2000


# ----------------------------------------------------------------------------------------------
# Подготовка
# ----------------------------------------------------------------------------------------------
def attach_gold(df: pd.DataFrame, ctx) -> pd.DataFrame:
    """Добавляет эталонную метку, страту и группу целевого комментария."""
    t = ctx.targets.set_index("id")
    out = df.copy()
    out["gold"] = out["target_id"].map(t["label"])
    out["stratum"] = out["target_id"].map(t["stratum"])
    out["share_toxic"] = out["target_id"].map(t["share_toxic"])
    out["group"] = out["target_id"].map(t["target_group"])
    out["ok"] = out["y"].notna() & out["error"].isna()
    return out


def _valid(df):
    return df[df["ok"]]


def cohens_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in (0, 1))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED, alpha: float = 0.05):
    """Бутстреп-интервал среднего (values — по одному числу на целевой комментарий)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ----------------------------------------------------------------------------------------------
# Качество классификации
# ----------------------------------------------------------------------------------------------
def confusion_metrics(y, gold) -> dict:
    y, gold = np.asarray(y, float), np.asarray(gold, float)
    tp = ((y == 1) & (gold == 1)).sum(); tn = ((y == 0) & (gold == 0)).sum()
    fp = ((y == 1) & (gold == 0)).sum(); fn = ((y == 0) & (gold == 1)).sum()
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return dict(n=int(n), accuracy=(tp + tn) / n if n else float("nan"), precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan"),
                fpr=fp / (fp + tn) if fp + tn else float("nan"), fnr=fn / (fn + tp) if fn + tp else float("nan"))


def classification_by_level(df: pd.DataFrame, kind: str = "batch") -> pd.DataFrame:
    """Доля TOXIC и качество по уровню токсичности соседей и группе целевых.

    Для спорных целевых (нет однозначной эталонной метки) считается только доля TOXIC и сравнение
    с долей разметчиков; точность/F1 — только для явных (clear)."""
    d = _valid(df[df["kind"] == kind])
    rows = []
    for (lvl, grp), g in d.groupby(["prop_toxic", "group"]):
        r = dict(prop_toxic=lvl, group=grp, n=len(g), share_toxic_pred=g["y"].mean(),
                 share_toxic_annotators=g["share_toxic"].mean())
        if grp == "clear":
            r.update({k: v for k, v in confusion_metrics(g["y"], g["gold"]).items() if k != "n"})
        rows.append(r)
    return pd.DataFrame(rows)


def unparsed_rates(df: pd.DataFrame, by: str = "prop_toxic") -> pd.DataFrame:
    """Доля запросов без разбираемого ответа (ошибка, обрезка, формат) — по уровню или позиции."""
    d = df.copy()
    d["bad"] = ~d["ok"]
    d["error_api"] = d["error"].notna()
    d["truncated"] = d["finish_reason"] == "length"
    return d.groupby(["kind", by], dropna=False)[["bad", "error_api", "truncated"]].mean().reset_index()


# ----------------------------------------------------------------------------------------------
# H1 / H4: влияние состава соседей
# ----------------------------------------------------------------------------------------------
def per_target(df: pd.DataFrame, level_col: str = "prop_toxic", kind: str = "batch") -> pd.DataFrame:
    """Доля TOXIC по (целевой × уровень): усреднение по повторам rep и позициям."""
    d = _valid(df[df["kind"] == kind])
    return d.groupby(["target_id", level_col])["y"].mean().unstack(level_col)


def effect(df: pd.DataFrame, lo: float = 0.0, hi: float = 1.0, groups: bool = True, kind: str = "batch") -> pd.DataFrame:
    """Эффект hi − lo (в п.п.) внутри целевого, бутстреп-ИК по целевым; по группам и в целом."""
    pt = per_target(df, kind=kind)
    if lo not in pt.columns or hi not in pt.columns:
        return pd.DataFrame()
    first = df.drop_duplicates("target_id").set_index("target_id")
    diff = (pt[hi] - pt[lo]).dropna()
    rows = []
    sets = {"all": diff.index}
    if groups:
        for g in ("clear", "ambiguous"):
            sets[g] = [i for i in diff.index if first.loc[i, "group"] == g]
        for s in ("clear_toxic", "clear_nontoxic"):
            sets[s] = [i for i in diff.index if first.loc[i, "stratum"] == s]
    for name, idx in sets.items():
        v = diff.loc[list(idx)].to_numpy() * 100
        m, a, b = bootstrap_ci(v)
        rows.append(dict(subset=name, n_targets=len(v), effect_pp=m, ci_low=a, ci_high=b,
                         ci_excludes_0=bool(a > 0 or b < 0), p_share_lo=pt.loc[list(idx), lo].mean() if len(v) else np.nan,
                         p_share_hi=pt.loc[list(idx), hi].mean() if len(v) else np.nan))
    return pd.DataFrame(rows)


def dose_response(df: pd.DataFrame) -> pd.DataFrame:
    """Доля TOXIC по всем уровням (для графика «доза—эффект»), ИК по целевым."""
    pt = per_target(df)
    rows = []
    for lvl in pt.columns:
        m, a, b = bootstrap_ci(pt[lvl].dropna().to_numpy() * 100)
        rows.append(dict(prop_toxic=lvl, share_toxic_pct=m, ci_low=a, ci_high=b, n_targets=int(pt[lvl].notna().sum())))
    return pd.DataFrame(rows)


def mcnemar_exact(df: pd.DataFrame, lo: float = 0.0, hi: float = 1.0) -> dict:
    """Точный тест Макнемара: у каждого целевого — мажоритарный ответ при lo и при hi."""
    pt = per_target(df)
    if lo not in pt.columns or hi not in pt.columns:
        return {}
    a = (pt[lo] >= 0.5).astype(int); b = (pt[hi] >= 0.5).astype(int)
    n01 = int(((a == 0) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(toxic_at_lo_only=n10, toxic_at_hi_only=n01, p_value=float(p), n=len(pt))


def gee_level(df: pd.DataFrame, family: str = "logit", extra: str = "") -> dict:
    """GEE: y ~ prop_toxic [+ extra], обменная корреляция, кластеры — целевые."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = _valid(df[df["kind"] == "batch"]).copy()
    d["y"] = d["y"].astype(float)
    if d["y"].nunique() < 2 or d["prop_toxic"].nunique() < 2:
        return {}
    fam = sm.families.Binomial() if family == "logit" else sm.families.Gaussian()
    f = "y ~ prop_toxic" + (f" + {extra}" if extra else "")
    try:
        m = smf.gee(f, "target_id", d, family=fam, cov_struct=sm.cov_struct.Exchangeable()).fit()
    except Exception as e:  # вырожденные данные
        return dict(error=f"{type(e).__name__}: {e}")
    ci = m.conf_int().loc["prop_toxic"]
    return dict(family=family, coef=float(m.params["prop_toxic"]), se=float(m.bse["prop_toxic"]),
                ci_low=float(ci[0]), ci_high=float(ci[1]), p_value=float(m.pvalues["prop_toxic"]),
                odds_ratio_0_to_100=float(np.exp(m.params["prop_toxic"])) if family == "logit" else None,
                n_obs=int(len(d)), n_clusters=int(d["target_id"].nunique()))


# ----------------------------------------------------------------------------------------------
# H5 / позиция
# ----------------------------------------------------------------------------------------------
def share_by_position(df: pd.DataFrame) -> pd.DataFrame:
    d = _valid(df[df["kind"] == "batch"])
    rows = []
    for (lvl, pos), g in d.groupby(["prop_toxic", "target_pos"]):
        pt = g.groupby("target_id")["y"].mean()
        m, a, b = bootstrap_ci(pt.to_numpy() * 100)
        rows.append(dict(prop_toxic=lvl, target_pos=pos, share_toxic_pct=m, ci_low=a, ci_high=b, n_targets=len(pt)))
    return pd.DataFrame(rows)


def position_slope(df: pd.DataFrame, level: float | None = None) -> dict:
    """Наклон доли TOXIC по позиции целевого (GEE, логит). level — фильтр по составу соседей."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = _valid(df[df["kind"] == "batch"]).copy()
    if level is not None:
        d = d[d["prop_toxic"] == level]
    d["y"] = d["y"].astype(float)
    if d["y"].nunique() < 2 or d["target_pos"].nunique() < 2:
        return {}
    m = smf.gee("y ~ target_pos", "target_id", d, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = m.conf_int().loc["target_pos"]
    return dict(level=level, slope_logit_per_pos=float(m.params["target_pos"]), ci_low=float(ci[0]), ci_high=float(ci[1]),
                p_value=float(m.pvalues["target_pos"]), n_obs=len(d))


def slope_x_position_wald(df: pd.DataFrame) -> dict:
    """Тест «наклон по составу зависит от позиции» (взаимодействие prop_toxic × target_pos), Wald на GEE."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = _valid(df[df["kind"] == "batch"]).copy()
    d["y"] = d["y"].astype(float)
    if d["y"].nunique() < 2 or d["target_pos"].nunique() < 2 or d["prop_toxic"].nunique() < 2:
        return {}
    m = smf.gee("y ~ prop_toxic * target_pos", "target_id", d, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    name = "prop_toxic:target_pos"
    return dict(coef=float(m.params[name]), se=float(m.bse[name]), p_value=float(m.pvalues[name]), n_obs=len(d))


# ----------------------------------------------------------------------------------------------
# Шум, температура, неуверенность
# ----------------------------------------------------------------------------------------------
def noise(df: pd.DataFrame, kind: str = "batch", samples=(0, 1, 2)) -> dict:
    """Шум модели: доля запросов, где ответ различается между независимыми повторами, и κ Коэна."""
    d = _valid(df[(df["kind"] == kind) & (df["sample"].isin(samples))])
    w = d.pivot_table(index="id", columns="sample", values="y", aggfunc="first").dropna()
    if w.shape[1] < 2:
        return {}
    vals = w.to_numpy()
    changed = float((vals != vals[:, [0]]).any(axis=1).mean())
    kappas = [cohens_kappa(vals[:, 0], vals[:, j]) for j in range(1, vals.shape[1])]
    return dict(kind=kind, n_items=len(w), share_changed=changed, mean_kappa_vs_first=float(np.nanmean(kappas)))


def temperature_check(df_t0: pd.DataFrame, df_t1: pd.DataFrame) -> dict:
    """Сравнение T=0 и T=1 для одиночных запросов: насколько температура сама меняет ответы."""
    a = _valid(df_t0[df_t0["kind"] == "single"]).groupby("target_id")["y"].mean()
    b = _valid(df_t1[df_t1["kind"] == "single"]).groupby("target_id")["y"].mean()
    j = pd.concat([a.rename("t0"), b.rename("t1")], axis=1).dropna()
    return dict(n=len(j), share_toxic_t0=float(j.t0.mean()), share_toxic_t1=float(j.t1.mean()),
                flip_rate=float(((j.t0 >= 0.5) != (j.t1 >= 0.5)).mean()))


def uncertainty(df_unc: pd.DataFrame) -> pd.Series:
    """u = доля TOXIC среди 10 ответов при T=1 на одиночный комментарий. 0.5 — максимальная неуверенность."""
    d = _valid(df_unc[df_unc["kind"] == "single"])
    return d.groupby("target_id")["y"].mean().rename("u")


def effect_by_uncertainty(df: pd.DataFrame, u: pd.Series, bins=(0, 0.1, 0.9, 1.0001)) -> pd.DataFrame:
    """H2 через неуверенность модели: эффект 100 % − 0 % отдельно для уверенных и неуверенных целевых."""
    pt = per_target(df)
    if 0.0 not in pt.columns or 1.0 not in pt.columns:
        return pd.DataFrame()
    diff = ((pt[1.0] - pt[0.0]) * 100).dropna().to_frame("effect").join(u, how="inner")
    diff["u_bin"] = pd.cut(diff["u"], bins=list(bins), right=False, labels=["u<0.1 (стабильно NOT)", "0.1≤u<0.9 (неуверен)", "u≥0.9 (стабильно TOXIC)"][: len(bins) - 1])
    rows = []
    for k, g in diff.groupby("u_bin", observed=True):
        m, a, b = bootstrap_ci(g["effect"].to_numpy())
        rows.append(dict(u_bin=k, n_targets=len(g), effect_pp=m, ci_low=a, ci_high=b, exploratory=True))
    return pd.DataFrame(rows)


def h2_ambiguous_vs_clear(df: pd.DataFrame) -> dict:
    """H2 по разметке: разность эффектов (ambiguous − clear) с бутстреп-ИК. Разные страты — сравнение между группами."""
    pt = per_target(df)
    if 0.0 not in pt.columns or 1.0 not in pt.columns:
        return {}
    first = df.drop_duplicates("target_id").set_index("target_id")
    diff = ((pt[1.0] - pt[0.0]) * 100).dropna()
    g = first.loc[diff.index, "group"]
    a, c = diff[g == "ambiguous"].to_numpy(), diff[g == "clear"].to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    boots = [rng.choice(a, len(a)).mean() - rng.choice(c, len(c)).mean() for _ in range(N_BOOT)]
    d = a.mean() - c.mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return dict(effect_ambiguous=float(a.mean()), effect_clear=float(c.mean()), diff=float(d), ci_low=float(lo),
                ci_high=float(hi), ci_excludes_0=bool(lo > 0 or hi < 0))


# ----------------------------------------------------------------------------------------------
# Наблюдательные анализы (exploratory): нормализация, серии, точность по соседям
# ----------------------------------------------------------------------------------------------
def normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Сколько TOXIC модель ставит во всём батче (по каждому тексту) против истинной доли токсичных соседей."""
    d = _valid(df[df["kind"] == "batch"]).copy()
    d["n_pred_toxic"] = d["labels"].apply(lambda l: sum(1 for v in l if v == 1))
    d["n_parsed"] = d["labels"].apply(lambda l: sum(1 for v in l if not pd.isna(v)))
    d["pred_share"] = d["n_pred_toxic"] / d["n_parsed"]
    d["true_share"] = (d["n_toxic"] + d["gold"]) / d["n_items"]
    out = d.groupby("prop_toxic")[["true_share", "pred_share"]].mean().reset_index()
    out["exploratory"] = True
    return out


def neighbor_accuracy(df: pd.DataFrame, ctx) -> pd.DataFrame:
    """Точность разметки соседей (не целевого) по уровню состава: показывает, не ломается ли сама разметка."""
    d = _valid(df[df["kind"] == "batch"])
    rows = []
    for r in d.itertuples():
        for pos, (i, l) in enumerate(zip(r.item_ids, r.labels), start=1):
            if pos != r.target_pos and not pd.isna(l):
                rows.append((r.prop_toxic, int(l == int(ctx.gold[i])), int(ctx.gold[i])))
    t = pd.DataFrame(rows, columns=["prop_toxic", "correct", "gold"])
    return t.groupby(["prop_toxic", "gold"])["correct"].mean().unstack("gold").rename(columns={0: "acc_nontoxic", 1: "acc_toxic"}).reset_index()


def run_length_before_target(df: pd.DataFrame) -> pd.DataFrame:
    """Наблюдательно: длина серии одинаковых ВЫДАННЫХ моделью меток прямо перед целевым и доля TOXIC у целевого."""
    d = _valid(df[(df["kind"] == "batch") & (df["target_pos"] > 1)])
    rows = []
    for r in d.itertuples():
        prev = [r.labels[i] for i in range(r.target_pos - 1)]
        if any(pd.isna(v) for v in prev):
            continue
        last, n = prev[-1], 0
        for v in reversed(prev):
            if v == last:
                n += 1
            else:
                break
        rows.append((int(last), n, int(r.y)))
    t = pd.DataFrame(rows, columns=["last_label", "run_len", "y"])
    out = t.groupby(["last_label", "run_len"])["y"].agg(["mean", "size"]).reset_index().rename(columns={"mean": "share_toxic_target", "size": "n"})
    out["exploratory"] = True
    return out


def singles_vs_gold(df: pd.DataFrame) -> dict:
    """Качество одиночной классификации на явных целевых + завышение токсичности на всех."""
    d = _valid(df[(df["kind"] == "single") & (df["sample"] == C.SAMPLE_BASE)])
    clear = d[d["group"] == "clear"]
    r = confusion_metrics(clear["y"], clear["gold"])
    r.update(share_toxic_pred_all=float(d["y"].mean()), share_toxic_gold_all=float(d["share_toxic"].mean()))
    return r


def effect_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Эффект 100 % − 0 % отдельно для каждой позиции целевого (h1c)."""
    rows = []
    for pos, g in df[df["kind"] == "batch"].groupby("target_pos"):
        e = effect(g, groups=False)
        if len(e):
            rows.append(dict(target_pos=int(pos), **e.iloc[0].drop("subset").to_dict()))
    return pd.DataFrame(rows)


def effect_pos1_vs_rest(df: pd.DataFrame) -> dict:
    """Эффект на позиции 1 (соседи только после целевого) и на позициях 2–9 (соседи есть и до него)."""
    b = df[df["kind"] == "batch"]
    out = {}
    for name, g in (("pos1", b[b["target_pos"] == 1]), ("pos2_9", b[b["target_pos"] > 1])):
        e = effect(g, groups=False)
        if len(e):
            out[name] = e.iloc[0].drop("subset").to_dict()
    return out
