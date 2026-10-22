"""Графики для статьи (подписи на русском). Читают таблицы из reports/tables (после `python -m ctxbias report`)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _read(model_key, name):
    p = C.TABLE_DIR / f"{model_key}_{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def dose_response(model_key):
    d = _read(model_key, "h1_dose_response")
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = d["prop_toxic"] * 100
    ax.errorbar(x, d["share_toxic_pct"], yerr=[d["share_toxic_pct"] - d["ci_low"], d["ci_high"] - d["share_toxic_pct"]], marker="o", capsize=3)
    ax.set(xlabel="Доля токсичных соседей, %", ylabel="Доля ответов TOXIC у целевого, %",
           title=f"{model_key}: зависимость от состава соседей")
    fig.tight_layout()
    return _save(fig, f"{model_key}_dose_response")


def position(model_key):
    d = _read(model_key, "h1c_share_by_position")
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for lvl, g in d.groupby("prop_toxic"):
        ax.errorbar(g["target_pos"], g["share_toxic_pct"], yerr=[g["share_toxic_pct"] - g["ci_low"], g["ci_high"] - g["share_toxic_pct"]],
                    marker="o", capsize=2, label=f"соседи токсичны на {lvl:.0%}")
    ax.set(xlabel="Позиция целевого в списке", ylabel="Доля TOXIC, %", title=f"{model_key}: позиция × состав")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, f"{model_key}_position")


def effect_groups(model_key):
    d = _read(model_key, "h1_effect")
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3))
    y = range(len(d))
    ax.errorbar(d["effect_pp"], list(y), xerr=[d["effect_pp"] - d["ci_low"], d["ci_high"] - d["effect_pp"]], fmt="o", capsize=3)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(list(y), d["subset"])
    ax.set(xlabel="Эффект: 100 % − 0 % токсичных соседей, п.п.", title=f"{model_key}: эффект по группам")
    fig.tight_layout()
    return _save(fig, f"{model_key}_effect_groups")


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_all(model_key):
    return [p for p in (dose_response(model_key), position(model_key), effect_groups(model_key)) if p]


_NAMES = {"gigachat_lite": "GigaChat-2 (Lite)", "gigachat_ultra": "GigaChat-3 Ultra", "deepseek": "DeepSeek (flash)", "openai": "GPT-5.6 terra", "claude": "Claude Sonnet 5.5"}
_COLORS = {"gigachat_lite": "#1b9e77", "gigachat_ultra": "#66a61e", "deepseek": "#d95f02", "openai": "#7570b3", "claude": "#e7298a"}


def compare_effects(models=("gigachat_lite", "gigachat_ultra", "deepseek", "openai", "claude")):
    """Лес-график: эффект 100 % − 0 % по моделям для основного эксперимента (h1) и эксперимента про позицию (h1c)."""
    p = C.TABLE_DIR / "comparison.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).set_index("model").loc[[m for m in models if m in pd.read_csv(p)["model"].values]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (col, lab, off) in enumerate((("h1", "основной (h1)", -0.15), ("h1c", "с перебором позиций (h1c)", 0.15))):
        y = [k + off for k in range(len(d))]
        ax.errorbar(d[f"{col}_effect"] if col == "h1" else d["h1c_effect"], y,
                    xerr=[(d[f"{col}_effect"] if col == "h1" else d["h1c_effect"]) - d[f"{col}_ci_low"],
                          d[f"{col}_ci_high"] - (d[f"{col}_effect"] if col == "h1" else d["h1c_effect"])],
                    fmt="o" if i == 0 else "s", capsize=3, label=lab)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(d)), [_NAMES[m] for m in d.index]); ax.invert_yaxis()
    ax.set(xlabel="Эффект: 100 % − 0 % токсичных соседей, п.п. (95 % ИК)", title="Контраст: токсичные соседи → реже TOXIC у целевого")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    return _save(fig, "comparison_effect")


def compare_positions(models=("gigachat_lite", "gigachat_ultra", "deepseek", "openai", "claude")):
    """Эффект состава соседей по позиции целевого в списке (h1c), все модели."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ok = False
    for m in models:
        p = C.TABLE_DIR / f"{m}_h1c_effect_by_position.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        ax.errorbar(d["target_pos"] + (list(models).index(m) - 2) * 0.06, d["effect_pp"],
                    yerr=[d["effect_pp"] - d["ci_low"], d["ci_high"] - d["effect_pp"]], marker="o", capsize=2,
                    color=_COLORS[m], label=_NAMES[m], lw=1.5)
        ok = True
    if not ok:
        return None
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(range(1, 10))
    ax.set(xlabel="Позиция целевого в списке из 9", ylabel="Эффект 100 % − 0 %, п.п.", title="Эффект зависит от позиции целевого")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "comparison_position")
