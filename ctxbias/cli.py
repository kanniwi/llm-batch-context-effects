"""Командная строка:  python -m ctxbias <команда> ...

  data                      скачать Jigsaw (нужен Kaggle/HF-доступ) и построить data/processed
  plans                     проверить, что планы батчей воспроизводятся из данных (детерминированность)
  budget  --model KEY       сколько запросов нужно и сколько уже в кэше (бесплатно)
  run EXP --model KEY       выполнить эксперимент (h1 | h1c | repeats | uncertainty | h3)
  report  --model KEY       посчитать метрики и таблицы из сохранённых результатов
  models                    список зарегистрированных моделей и наличия ключей
  ping    --model KEY       3 пробных запроса (копейки): ключ, имя модели, формат ответа, расход токенов
"""
from __future__ import annotations

import argparse
import os
import sys

from . import config as C


def _ctx():
    from .experiments import Context
    return Context.load()


def main(argv=None) -> int:
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
    p = argparse.ArgumentParser(prog="ctxbias", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data"); sub.add_parser("plans"); sub.add_parser("models")
    sub.add_parser("ping").add_argument("--model", required=True, choices=sorted(C.MODELS))
    for name in ("budget", "run", "report"):
        sp = sub.add_parser(name)
        if name == "run":
            sp.add_argument("experiment")
        sp.add_argument("--model", required=True, choices=sorted(C.MODELS))
        if name != "report":
            sp.add_argument("--experiment-name", dest="exp_opt")
            sp.add_argument("--samples", type=int, nargs="*", help="переопределить номера независимых повторов")
    sub.choices["budget"].add_argument("experiments", nargs="*", help="по умолчанию — все")
    a = p.parse_args(argv)

    if a.cmd == "models":
        for k, s in C.MODELS.items():
            key = "—" if not s.api_key_env else ("есть" if os.getenv(s.api_key_env) else "НЕТ ключа")
            print(f"{k:20s} {s.provider:10s} {s.model:24s} ключ {s.api_key_env or '-'}: {key}")
        return 0
    if a.cmd == "ping":
        from . import llm
        from .prompts import SYSTEM, build_user_prompt, parse_labels
        texts = ["You are an idiot and everyone knows it.", "Thanks for the clear explanation, this helped a lot.",
                 "I disagree with the mayor, but the budget numbers look correct."]
        for i, t in enumerate(texts, 1):
            r = llm.ask(a.model, SYSTEM["en"], build_user_prompt([t]), 0.0, C.MAX_TOKENS_SINGLE, sample=900 + i)
            labels, ok = parse_labels(r["text"], 1)
            print(f"{i}) ответ: {r['text']!r} | разобрано: {ok} | finish: {r.get('finish_reason')} | модель: {r.get('served_model')} "
                  f"| токены: {r.get('usage')} | темп. применена: {r.get('temperature_applied', True)} | ошибка: {r.get('error')}")
            if r.get("error"):
                break
        return 0
    if a.cmd == "data":
        from . import data
        t, n = data.build_and_save()
        print(f"целевых: {len(t)}, соседей: {len(n)}")
        return 0
    if a.cmd == "plans":
        from . import plans
        ctx = _ctx()
        h1 = plans.build_h1_plan(ctx.targets, ctx.neighbors)
        h1c = plans.build_h1c_plan(ctx.targets, ctx.neighbors)
        print(f"h1: {len(h1)} батчей; h1c: {len(h1c)} батчей; repeats: {len(plans.repeat_subset(h1))} батчей")
        return 0
    from . import runner
    from .experiments import EXPERIMENTS
    samples = tuple(a.samples) if getattr(a, "samples", None) else None
    ctx = _ctx()
    if a.cmd == "budget":
        total_new = 0
        for e in (a.experiments or EXPERIMENTS):
            if e == "h3" and ctx.neutral is None:
                print("h3: нужен пул нейтральных соседей (ещё не построен)")
                continue
            r = runner.count_requests(e, a.model, ctx, samples)
            total_new += r["remaining"]
            print(f"{e:12s} всего {r['total']:6d}  в кэше {r['cached']:6d}  осталось {r['remaining']:6d}")
        print(f"итого новых запросов: {total_new}")
        return 0
    if a.cmd == "run":
        df = runner.run(a.experiment, a.model, samples, ctx)
        print(f"готово: {len(df)} запросов, ошибок {int(df['error'].notna().sum())}, не разобрано {int((~df['parse_ok'] & df['error'].isna()).sum())}")
        return 0
    if a.cmd == "report":
        from . import report
        S = report.build(a.model, ctx)
        print(f"таблицы: {C.TABLE_DIR}; ключей в сводке: {len(S)}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
