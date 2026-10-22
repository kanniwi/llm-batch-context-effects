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


# ---------------------------------------------------------------------------------------------
# H3: условия подачи (ЧЕРНОВИК — состав условий согласовать с преподавателем)
# ---------------------------------------------------------------------------------------------
H3_CONDITIONS = ("standard", "target_only", "labels_true", "labels_flipped", "labels_only")

_DEF = ("You are a content moderator. A comment is TOXIC if it is rude, disrespectful, or unreasonable "
        "and likely to make someone leave a discussion. Otherwise it is NOT_TOXIC.\n")
SYSTEM_H3_TARGET = (_DEF +
                    "You will receive numbered comments. Judge ONLY the comment you are asked about, "
                    "independently and based on its own content.\n"
                    "Reply with exactly one line in the form `<number>: TOXIC` or `<number>: NOT_TOXIC`. "
                    "Do not write anything else.")
SYSTEM_H3_LABELED = (_DEF +
                     "You will receive numbered comments. Some of them already carry a label from another moderator, "
                     "marked `Label:`. Judge ONLY the comment you are asked about, independently and based on its own content.\n"
                     "Reply with exactly one line in the form `<number>: TOXIC` or `<number>: NOT_TOXIC`. "
                     "Do not write anything else.")
MASK = "[text hidden]"


def build_h3_prompt(condition: str, texts: list[str], gold: list[int | None], target_pos: int) -> tuple[str, str]:
    """Возвращает (system, user). gold[i] — человеческая разметка i-го текста (для целевого игнорируется).

    standard       — как в основном эксперименте: модель размечает все тексты;
    target_only    — все тексты показаны, метку просим только у целевого (своих меток соседей модель не порождает);
    labels_true    — у соседей указаны верные метки (из разметки людей), метку просим у целевого;
    labels_flipped — у соседей указаны инвертированные метки;
    labels_only    — тексты соседей скрыты, указаны только их верные метки."""
    n = len(texts)
    if condition == "standard":
        return SYSTEM["en"], build_user_prompt(texts)
    lab = lambda g: "TOXIC" if g == 1 else "NOT_TOXIC"
    lines = []
    for i, (t, g) in enumerate(zip(texts, gold), start=1):
        if i == target_pos or condition == "target_only":
            lines.append(f"[{i}] {t}")
        elif condition == "labels_true":
            lines.append(f"[{i}] {t}\n    Label: {lab(g)}")
        elif condition == "labels_flipped":
            lines.append(f"[{i}] {t}\n    Label: {lab(1 - g)}")
        elif condition == "labels_only":
            lines.append(f"[{i}] {MASK}\n    Label: {lab(g)}")
        else:
            raise ValueError(f"неизвестное условие H3: {condition}")
    system = SYSTEM_H3_TARGET if condition == "target_only" else SYSTEM_H3_LABELED
    user = "Comments:\n\n" + "\n".join(lines) + f"\n\nAnswer only for comment [{target_pos}]."
    return system, user
