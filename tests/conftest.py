"""Тесты работают в изолированной папке и только с имитацией модели (provider=mock): запросов к API нет."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="ctxbias_test_")
os.environ["CTXBIAS_ROOT"] = _TMP          # до импорта ctxbias: кэш и результаты пишутся во временную папку

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def make_synthetic(n_targets=24, n_neigh=300, seed=0):
    rng = np.random.default_rng(seed)

    def mk(i, label, art):
        toks = int(rng.integers(20, 40))
        words = ["word"] * toks
        if label:
            words[0] = "TOXWORD"
        return dict(id=i, text=" ".join(words) + f" {i}", label=label, n_tokens=toks, article_id=art)

    nb = [mk(10_000 + i, i % 2, i % 50) for i in range(n_neigh)]
    tg = []
    for i in range(n_targets):
        label = i % 2
        r = mk(i, label, 1000 + i)
        r.update(stratum="clear_toxic" if label else "clear_nontoxic", target_group="clear", share_toxic=float(label))
        tg.append(r)
    for i in range(n_targets, n_targets + 12):
        r = mk(i, i % 2, 1000 + i)
        r.update(stratum="ambiguous", target_group="ambiguous", share_toxic=0.5)
        tg.append(r)
    return pd.DataFrame(tg), pd.DataFrame(nb)


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic()
