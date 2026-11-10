"""Сквозная проверка на имитации модели: запуск → кэш → отчёт; повторный запуск не делает новых запросов."""
import pandas as pd

from ctxbias import config as C
from ctxbias import data, metrics as M, report, runner
from ctxbias.experiments import Context


def _ctx(synthetic):
    t, n = synthetic
    return Context(t, n, None)


def test_end_to_end(synthetic, monkeypatch):
    t, n = synthetic
    # h1c берёт фиксированные 100 целевых: для маленького набора подменяем выборку
    from ctxbias import plans
    monkeypatch.setattr(plans, "sample_h1c_targets", lambda targets: targets.iloc[:6])
    ctx = _ctx(synthetic)
    b = runner.count_requests("repeats", "mock", ctx)
    assert b["cached"] == 0 and b["remaining"] == b["total"]
    df = runner.run("h1", "mock", ctx=ctx)
    assert len(df) == len(t) * 15 + len(t) + len(n)
    assert df["error"].isna().all() and df["parse_ok"].all()
    assert (C.DATA_RES / "h1_mock.manifest.json").exists()
    # повторный запуск полностью из кэша
    assert runner.count_requests("h1", "mock", ctx)["remaining"] == 0
    # метрики работают и дают контраст/ассимиляцию без падений
    d = M.attach_gold(df[df["sample"] == 0], ctx)
    eff = M.effect(d)
    assert set(eff.subset) >= {"all", "clear", "ambiguous"}
    assert M.gee_level(d, "logit") and M.mcnemar_exact(d)
    # на имитации с ассимиляцией (+1.2*(доля-0.5)) эффект не должен быть отрицательным
    assert eff[eff.subset == "all"].effect_pp.iloc[0] >= 0
    runner.run("uncertainty", "mock", ctx=ctx)
    runner.run("repeats", "mock", ctx=ctx)
    runner.run("h1c", "mock", ctx=ctx)
    S = report.build("mock", ctx)
    assert "h1_effect" in S and "noise_batch" in S and S["noise_batch"]["share_changed"] == 0.0


def test_h3_and_report(synthetic, monkeypatch):
    from ctxbias import plans
    monkeypatch.setattr(plans, "sample_h1c_targets", lambda targets: targets.iloc[:6])
    ctx = _ctx(synthetic)
    plan = ctx.h3_plan
    assert len(plan) == 12 * len(ctx.targets)
    for _, group in plan.groupby(["target_id", "target_pos", "n_toxic"]):
        assert len(group) == 3 and len({tuple(ids) for ids in group.item_ids}) == 1

    full = runner.run("h3", "mock", ctx=ctx)
    assert full["parse_ok"].all()
    assert runner.count_requests("h3", "mock", ctx)["remaining"] == 0
    assert report.build("mock", ctx)["h3_primary"]["n_targets"] == len(ctx.targets)

    # Искусственный пример: метки меняют ответ у всех, тексты не меняют.
    full["y"] = full["condition"].map(dict(A=0, B=1, C=0, D=1, E=0, F=0))
    full["ok"] = True
    full.loc[full.index[full["condition"] == "A"][0], "ok"] = False
    effects = M.h3_effects(full)
    main = effects.query("scope == 'all' and contrast == 'abs_labels_minus_abs_texts'").iloc[0]
    assert main.n_targets == len(ctx.targets) - 1
    assert (main.effect_pp, main.ci_low, main.ci_high) == (100, 100, 100)
