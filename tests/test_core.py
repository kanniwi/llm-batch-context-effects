import hashlib
import json

import numpy as np
import pandas as pd

from ctxbias import config as C
from ctxbias import llm, metrics as M, plans
from ctxbias.prompts import build_user_prompt, parse_labels


def test_parse_labels_variants():
    assert parse_labels("1: TOXIC\n2: NOT_TOXIC", 2) == ([1, 0], True)
    assert parse_labels("[1]: NOT TOXIC\n[2]: Toxic", 2) == ([0, 1], True)
    labels, ok = parse_labels("1: TOXIC", 2)
    assert labels == [1, None] and not ok
    assert parse_labels("", 3) == ([None] * 3, False)


def test_prompt_format():
    assert build_user_prompt(["a", "b"]) == "Comments:\n\n[1] a\n[2] b\n\nAnswer for all 2 comments."
    assert "1 comment." in build_user_prompt(["a"])


def test_request_key_is_stable():
    """Ключ кэша не должен меняться: иначе пропадут ответы, уже полученные за деньги."""
    spec = C.MODELS["deepseek"]
    raw = json.dumps([spec.model, "S", "U", 0.0, 16, 0], ensure_ascii=False)
    assert llm.request_key(spec, "S", "U", 0, 16, 0) == hashlib.sha256(raw.encode()).hexdigest()
    assert llm.request_key(spec, "S", "U", 0, 16, 1) != llm.request_key(spec, "S", "U", 0, 16, 0)


def test_plan_invariants(synthetic):
    t, n = synthetic
    p = plans.build_h1_plan(t, n)
    assert len(p) == len(t) * 5 * 3
    gold = pd.concat([t, n]).set_index("id")["label"]
    art = pd.concat([t, n]).set_index("id")["article_id"]
    for r in p.itertuples():
        ids = list(r.item_ids)
        assert len(ids) == 9 and len(set(ids)) == 9
        assert ids[r.target_pos - 1] == r.target_id
        neigh = [i for i in ids if i != r.target_id]
        assert sum(gold[i] for i in neigh) == r.n_toxic
        assert all(art[i] != art[r.target_id] for i in neigh)
    # детерминированность
    assert plans.build_h1_plan(t, n).equals(p)


def test_nested_neighbor_sets(synthetic):
    t, n = synthetic
    p = plans.build_h1_plan(t, n)
    g = p[(p.target_id == t.id.iloc[0]) & (p.rep == 0)].set_index("n_toxic")["item_ids"]
    gold = n.set_index("id")["label"]
    tox = lambda k: {i for i in g[k] if i in gold.index and gold[i] == 1}
    assert tox(2) <= tox(4) <= tox(8)


def test_noise_and_kappa():
    assert M.cohens_kappa([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0
    assert M.cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0]) == -1.0


def test_bootstrap_ci_contains_mean():
    m, a, b = M.bootstrap_ci(np.r_[np.ones(50), np.zeros(50)])
    assert a <= m <= b and abs(m - 0.5) < 1e-9


def test_effect_sign_on_toy():
    rows = []
    for tid in range(40):
        for lvl in (0.0, 1.0):
            for rep in range(3):
                rows.append(dict(target_id=tid, kind="batch", prop_toxic=lvl, y=1.0 if lvl == 0 else 0.0, ok=True,
                                 group="clear", stratum="clear_toxic", rep=rep))
    e = M.effect(pd.DataFrame(rows))
    r = e[e.subset == "all"].iloc[0]
    assert r.effect_pp == -100 and r.ci_excludes_0
