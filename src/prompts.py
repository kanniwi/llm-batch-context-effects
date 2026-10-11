import math
import re

SYSTEM_EN = (
    "You are a content moderator. A comment is TOXIC if it is rude, disrespectful, or unreasonable "
    "and likely to make someone leave a discussion. Otherwise it is NOT_TOXIC.\n"
    "You will receive one or more numbered comments. Judge each comment independently, "
    "based only on its own content.\n"
    "Reply with exactly one line per comment, in the same order, in the form "
    "`<number>: TOXIC` or `<number>: NOT_TOXIC`. Do not write anything else."
)

SYSTEM = {"en": SYSTEM_EN}


def build_user_prompt(texts: list[str]) -> str:
    n = len(texts)
    body = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts, start=1))
    noun = "comment" if n == 1 else "comments"
    return f"Comments:\n\n{body}\n\nAnswer for all {n} {noun}."


LINE_RE = re.compile(r"^\s*\[?(\d+)\]?\s*[:.)\-]\s*\**\s*(NOT[_ ]?TOXIC|NON[_ \-]?TOXIC|TOXIC)\b", re.I)


def parse_labels(text: str, n: int):
    labels = [None] * n
    seen = 0
    for line in (text or "").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        k = int(m.group(1))
        if 1 <= k <= n and labels[k - 1] is None:
            labels[k - 1] = 0 if m.group(2).upper().startswith(("NOT", "NON")) else 1
            seen += 1
    return labels, seen == n


_PREFIX_RE = re.compile(r"^\s*\[?\d+\]?\s*[:.)\-]?\s*\**\s*$")
_LABEL_START_RE = re.compile(r"^\s*\[?(\d+)\]?\s*[:.)\-]\s*\**\s*([A-Za-z_]+)")


def _kind(word: str):
    w = word.upper()
    if w.startswith(("NOT", "NON", "NO")):
        return "not"
    if w.startswith(("TOX", "TO", "T")):
        return "tox"
    return None


def p_toxic_from_logprobs(tokens, n: int):
    probs = [None] * n
    line = ""
    for tok, alts in tokens:
        if "\n" in tok:
            line = tok.split("\n")[-1]
            continue
        new = line + tok
        m = _LABEL_START_RE.match(new)
        if m and _PREFIX_RE.match(line):
            k = int(m.group(1))
            if 1 <= k <= n and probs[k - 1] is None:
                p = {"tox": 0.0, "not": 0.0}
                for alt, lp in alts:
                    mm = _LABEL_START_RE.match(line + alt)
                    kind = _kind(mm.group(2)) if mm else None
                    if kind:
                        p[kind] += math.exp(lp)
                if p["tox"] + p["not"] > 0:
                    probs[k - 1] = p["tox"] / (p["tox"] + p["not"])
        line = new
    return probs
