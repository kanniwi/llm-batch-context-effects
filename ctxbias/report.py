"""Сборка таблиц и сводки по результатам одной модели (всё считается из data/results, без запросов к API)."""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from . import metrics as M
from .experiments import Context
from .runner import load_results


def _try(name):
    def deco(f):
        return f
    return deco


def _load(exp, mk, ctx):
    try:
        return M.attach_gold(load_results(exp, mk), ctx)
    except FileNotFoundError:
        return None


def build(model_key: str, ctx: Context | None = None) -> dict:
    """Считает все метрики для модели; пишет CSV в reports/tables и summary_<модель>.json. Возвращает summary."""
    ctx = ctx or Context.load()
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    S: dict = {"model_key": model_key}

    def save(df, name):
        if df is not None and len(df):
            df.to_csv(C.TABLE_DIR / f"{model_key}_{name}.csv", index=False, encoding="utf-8")

    h1 = _load("h1", model_key, ctx)
    if h1 is not None:
        b = h1[h1["sample"] == C.SAMPLE_BASE]
        S["h1_singles"] = M.singles_vs_gold(b)
        eff = M.effect(b); save(eff, "h1_effect"); S["h1_effect"] = eff.to_dict("records")
        save(M.dose_response(b), "h1_dose_response")
        save(M.classification_by_level(b), "h1_classification_by_level")
        save(M.classification_by_level_stratum(h1), "h1_classification_by_level_stratum")
        save(M.unparsed_rates(b), "h1_unparsed")
        S["h1_gee_logit"] = M.gee_level(b, "logit"); S["h1_gee_linear"] = M.gee_level(b, "linear")
        S["h1_mcnemar"] = M.mcnemar_exact(b); S["h2_ambiguous_vs_clear"] = M.h2_ambiguous_vs_clear(b)
        save(M.normalization(b), "h1_normalization_exploratory")
        save(M.neighbor_accuracy(b, ctx), "h1_neighbor_accuracy")

    h1c = _load("h1c", model_key, ctx)
    if h1c is not None:
        b = h1c[h1c["sample"] == C.SAMPLE_BASE]
        save(M.share_by_position(b), "h1c_share_by_position")
        save(M.effect_by_position(b), "h1c_effect_by_position")
        S["h1c_effect_pos1_vs_rest"] = M.effect_pos1_vs_rest(b)
        save(M.effect(b), "h1c_effect")
        S["h1c_position_drift_at_0pct"] = M.position_slope(b, 0.0)
        S["h1c_position_drift_at_100pct"] = M.position_slope(b, 1.0)
        S["h1c_slope_x_position"] = M.slope_x_position_wald(b)
        save(M.unparsed_rates(b, "target_pos"), "h1c_unparsed")
        save(M.run_length_before_target(b), "h1c_run_length_exploratory")

    rep = _load("repeats", model_key, ctx)
    if rep is not None:
        S["noise_batch"] = M.noise(rep, "batch"); S["noise_single"] = M.noise(rep, "single")

    unc = _load("uncertainty", model_key, ctx)
    if unc is not None:
        u = M.uncertainty(unc); save(u.reset_index(), "uncertainty_u")
        S["uncertainty_mean_u"] = float(u.mean())
        if h1 is not None:
            save(M.effect_by_uncertainty(h1[h1["sample"] == C.SAMPLE_BASE], u), "h2_effect_by_uncertainty_exploratory")
        if rep is not None:
            S["temperature_check"] = M.temperature_check(rep[rep["sample"] == C.SAMPLE_BASE], unc)

    (C.TABLE_DIR / f"summary_{model_key}.json").write_text(json.dumps(S, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return S


MODEL_ORDER = ("gigachat_lite", "gigachat_ultra", "deepseek", "openai", "claude")


def compare(models=MODEL_ORDER) -> pd.DataFrame:
    """Сводная таблица по моделям (reports/tables/comparison.csv) из summary_<модель>.json."""
    rows = []
    for mk in models:
        p = C.TABLE_DIR / f"summary_{mk}.json"
        if not p.exists():
            continue
        S = json.loads(p.read_text(encoding="utf-8"))
        eff = {r["subset"]: r for r in S.get("h1_effect", [])}
        h1c = pd.read_csv(C.TABLE_DIR / f"{mk}_h1c_effect.csv").set_index("subset") if (C.TABLE_DIR / f"{mk}_h1c_effect.csv").exists() else None
        pr = S.get("h1c_effect_pos1_vs_rest", {})
        h2 = S.get("h2_ambiguous_vs_clear", {})
        s1 = S.get("h1_singles", {})
        row = dict(model=mk,
                   h1_effect=eff["all"]["effect_pp"], h1_ci_low=eff["all"]["ci_low"], h1_ci_high=eff["all"]["ci_high"],
                   h1_clear=eff["clear"]["effect_pp"], h1_ambiguous=eff["ambiguous"]["effect_pp"],
                   h2_diff=h2.get("diff"), h2_ci_low=h2.get("ci_low"), h2_ci_high=h2.get("ci_high"),
                   h1c_effect=None if h1c is None else h1c.loc["all", "effect_pp"],
                   h1c_ci_low=None if h1c is None else h1c.loc["all", "ci_low"],
                   h1c_ci_high=None if h1c is None else h1c.loc["all", "ci_high"],
                   pos1=pr.get("pos1", {}).get("effect_pp"), pos1_ci_low=pr.get("pos1", {}).get("ci_low"), pos1_ci_high=pr.get("pos1", {}).get("ci_high"),
                   pos2_9=pr.get("pos2_9", {}).get("effect_pp"), pos2_9_ci_low=pr.get("pos2_9", {}).get("ci_low"), pos2_9_ci_high=pr.get("pos2_9", {}).get("ci_high"),
                   interaction_p=S.get("h1c_slope_x_position", {}).get("p_value"),
                   noise_batch=S.get("noise_batch", {}).get("share_changed"), noise_single=S.get("noise_single", {}).get("share_changed"),
                   single_f1=s1.get("f1"), single_fpr=s1.get("fpr"), share_toxic_pred=s1.get("share_toxic_pred_all"),
                   share_toxic_gold=s1.get("share_toxic_gold_all"))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLE_DIR / "comparison.csv", index=False, encoding="utf-8")
    return df


def classification_tables(models=MODEL_ORDER) -> str:
    """Пишет docs/classification_tables.md: качество по уровням и по трём группам целевых для всех моделей."""
    names = {"gigachat_lite": "GigaChat-2 (Lite)", "gigachat_ultra": "GigaChat-3 Ultra", "deepseek": "DeepSeek",
             "openai": "GPT-5.6 terra", "claude": "Claude Sonnet 5.5"}
    f = lambda v: "—" if pd.isna(v) else f"{v:.2f}"
    out = ["# Качество классификации по уровню токсичности соседей и по группам целевых\n",
           "Источник: основной эксперимент `h1` (300 целевых, ответы при запрошенной T=0), таблицы `reports/tables/<модель>_h1_classification_by_level_stratum.csv`.\n",
           "Определения: accuracy — доля верных ответов; F1 — по классу TOXIC; FPR — доля нетоксичных целевых, названных токсичными; "
           "FNR — доля токсичных целевых, названных нетоксичными. «Без соседей» — одиночный запрос. "
           "Уровень = доля токсичных среди 8 соседей.\n",
           "Группы: **clear (все явные)** — 75 токсичных + 75 нетоксичных, для них считаются все показатели. **clear_toxic** (только токсичные: определены accuracy = 1 − FNR и FNR), "
           "**clear_nontoxic** (только нетоксичные: accuracy = 1 − FPR и FPR). **ambiguous** (доля разметчиков 0,4–0,6): эталон условный "
           "(доля ≥ 0,5), по сути близкий к подбрасыванию монеты, показатели справочные.\n"]
    for mk in models:
        p = C.TABLE_DIR / f"{mk}_h1_classification_by_level_stratum.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        out.append(f"\n## {names.get(mk, mk)}\n")
        for st, title in (("clear", "Явные целевые (clear), 150"), ("clear_toxic", "Явно токсичные (clear_toxic), 75"),
                          ("clear_nontoxic", "Явно нетоксичные (clear_nontoxic), 75"), ("ambiguous", "Спорные (ambiguous), 150, условный эталон")):
            g = d[d["group"] == st]
            if not len(g):
                continue
            cols = {"clear": ("accuracy", "f1", "fpr", "fnr"), "clear_toxic": ("accuracy", "fnr"),
                    "clear_nontoxic": ("accuracy", "fpr"), "ambiguous": ("accuracy", "f1", "fpr", "fnr")}[st]
            hdr = {"accuracy": "accuracy", "f1": "F1", "fpr": "FPR", "fnr": "FNR"}
            out.append(f"\n**{title}**\n")
            out.append("| Доля токсичных соседей | доля TOXIC у модели | " + " | ".join(hdr[c] for c in cols) + " |")
            out.append("|---|---|" + "---|" * len(cols))
            for r in g.itertuples():
                lvl = r.level if r.level == "без соседей" else f"{float(r.level):.0%}"
                out.append(f"| {lvl} | {f(r.share_toxic_pred)} | " + " | ".join(f(getattr(r, c)) for c in cols) + " |")
    text = "\n".join(out) + "\n"
    (C.ROOT / "docs").mkdir(exist_ok=True)
    (C.ROOT / "docs" / "classification_tables.md").write_text(text, encoding="utf-8")
    return text
