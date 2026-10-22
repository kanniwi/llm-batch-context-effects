"""Единый клиент к моделям: провайдеры, дисковый кэш, повторы, ограничение по дневному лимиту.

Принципы воспроизводимости
* Любой ответ сохраняется в `data/cache/<ключ_модели>.jsonl`. Ключ кэша = sha256 от
  (идентификатор модели, system, user, температура, max_tokens, sample). Поэтому повторный запуск
  с теми же параметрами не обращается к API, а `sample` = 1, 2, ... даёт независимые повторы.
* Ответы, оборванные по длине (finish_reason == "length"), в кэш не попадают.
* Ошибки API в кэш не попадают.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

from .config import CACHE_DIR, MODELS, ROOT, ModelSpec

load_dotenv(ROOT / ".env", override=True)

_clients: dict = {}
_logprobs_supported: dict = {}
_daily_limit_hit: dict = {}   # ключ_модели -> текст ошибки, если исчерпан ДНЕВНОЙ лимит провайдера


class DailyLimitError(Exception):
    """Дневной лимит (запросов или токенов) исчерпан: повторять сегодня бесполезно."""


def _is_daily_limit(e) -> bool:
    msg = str(e).lower()
    fatal = ("invalid_api_key", "authenticationerror", "incorrect api key", "insufficient_quota", "permissiondenied",
             "billing_hard_limit", "credit balance is too low", "authentication_error", "invalid x-api-key",
             "permission_error", "error code: 401", "error code: 403")      # ключ/оплата: повторы бесполезны, останавливаемся сразу
    return "per day" in msg or "(tpd)" in msg or "(rpd)" in msg or any(w in msg for w in fatal)


def daily_limit_hit(model_key: str):
    return _daily_limit_hit.get(model_key)


def reset_daily_limit(model_key: str | None = None):
    _daily_limit_hit.clear() if model_key is None else _daily_limit_hit.pop(model_key, None)


# ---------------------------------------------------------------------------------------------
# Клиенты провайдеров
# ---------------------------------------------------------------------------------------------
def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _client(spec: ModelSpec):
    ck = (spec.provider, spec.base_url)
    if ck in _clients:
        return _clients[ck]
    if spec.api_key_env and not os.getenv(spec.api_key_env):
        raise RuntimeError(f"Нет ключа {spec.api_key_env} в файле .env")
    p = spec.provider
    if p in ("openai", "deepseek"):
        from openai import OpenAI
        kw = dict(api_key=os.environ[spec.api_key_env])
        if spec.base_url:
            kw["base_url"] = spec.base_url
        c = OpenAI(**kw)
    elif p == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=os.environ[spec.api_key_env])
    elif p == "gigachat":
        from gigachat import GigaChat
        verify = _env_flag("GIGACHAT_VERIFY_SSL", True)
        kw = dict(credentials=os.environ[spec.api_key_env], scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                  verify_ssl_certs=verify, profanity_check=False, timeout=120)
        if verify and os.getenv("GIGACHAT_CA_BUNDLE"):
            kw["ca_bundle_file"] = os.environ["GIGACHAT_CA_BUNDLE"]   # корневой сертификат НУЦ Минцифры
        c = GigaChat(**kw)
    else:
        c = None
    _clients[ck] = c
    return c


# ---------------------------------------------------------------------------------------------
# Вызовы провайдеров. Каждый возвращает dict(text, tokens, usage, finish_reason, served_model)
# ---------------------------------------------------------------------------------------------
def _parse_openai_response(r):
    ch = r.choices[0]
    tokens = None
    if getattr(ch, "logprobs", None) and ch.logprobs.content:
        tokens = [(t.token, [(a.token, a.logprob) for a in (t.top_logprobs or [])]) for t in ch.logprobs.content]
    usage = dict(input=r.usage.prompt_tokens, output=r.usage.completion_tokens) if r.usage else {}
    return dict(text=ch.message.content or "", tokens=tokens, usage=usage, finish_reason=ch.finish_reason,
                served_model=getattr(r, "model", None))


def _openai_kwargs(spec: ModelSpec, system, user, temperature, max_tokens):
    kw = dict(model=spec.model, temperature=temperature,
              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if spec.provider == "openai":
        kw["max_completion_tokens"] = max_tokens
        kw["reasoning_effort"] = "none"
    else:
        kw["max_tokens"] = max_tokens
        if spec.provider == "deepseek":
            kw["extra_body"] = {"thinking": {"type": "disabled"}}
    return kw


def _call_openai_compatible(spec, system, user, temperature, max_tokens):
    client = _client(spec)
    kw = _openai_kwargs(spec, system, user, temperature, max_tokens)
    want_lp = spec.logprobs and _logprobs_supported.get(spec.model, True)
    if want_lp:
        kw.update(logprobs=True, top_logprobs=10)
    try:
        return _parse_openai_response(client.chat.completions.create(**kw))
    except Exception as e:
        msg = str(e).lower()
        if want_lp and "logprob" in msg:
            _logprobs_supported[spec.model] = False
            kw.pop("logprobs"); kw.pop("top_logprobs")
            return _parse_openai_response(client.chat.completions.create(**kw))
        if kw.get("extra_body") and spec.provider == "deepseek" and any(
                w in msg for w in ("thinking", "reasoning", "effort")):
            kw.pop("extra_body")      # шлюз не принял параметр отключения рассуждений: повтор без него
            return _parse_openai_response(client.chat.completions.create(**kw))
        raise


_anthropic_temp_ok: dict = {}      # модель -> принимает ли API параметр temperature (в новых SDK его нет в сигнатуре)


def _call_anthropic(spec, system, user, temperature, max_tokens):
    """Новые модели Anthropic могут не принимать температуру. Пробуем передать её через extra_body; если API отказывает,
    повторяем без неё и помечаем ответ (`temperature_applied=False`): тогда запрос идёт при температуре по умолчанию."""
    client = _client(spec)
    kw = dict(model=spec.model, system=system, max_tokens=max_tokens, messages=[{"role": "user", "content": user}])
    applied = _anthropic_temp_ok.get(spec.model, True)
    if applied:
        try:
            r = client.messages.create(**kw, extra_body={"temperature": temperature})
        except Exception as e:
            if "temperature" in str(e).lower() and ("400" in str(e) or "invalid_request" in str(e).lower()
                                                   or "deprecated" in str(e).lower() or "not supported" in str(e).lower()):
                _anthropic_temp_ok[spec.model] = False
                applied = False
            else:
                raise
    if not applied:
        r = client.messages.create(**kw)
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return dict(text=text, tokens=None, finish_reason=r.stop_reason, served_model=getattr(r, "model", None),
                temperature_applied=applied, usage=dict(input=r.usage.input_tokens, output=r.usage.output_tokens))


def _call_gigachat(spec, system, user, temperature, max_tokens):
    from gigachat.models import Chat, Messages, MessagesRole
    chat = Chat(model=spec.model, max_tokens=max_tokens, profanity_check=False,
                temperature=max(temperature, 0.001),    # API не принимает ровно 0
                messages=[Messages(role=MessagesRole.SYSTEM, content=system),
                          Messages(role=MessagesRole.USER, content=user)])
    r = _client(spec).chat(chat)
    ch = r.choices[0]
    return dict(text=ch.message.content or "", tokens=None, finish_reason=ch.finish_reason,
                served_model=getattr(r, "model", None),
                usage=dict(input=r.usage.prompt_tokens, output=r.usage.completion_tokens))


def _call_mock(spec, system, user, temperature, max_tokens):
    """Детерминированная имитация модели для автотестов (токсичность = наличие слова TOXWORD)."""
    import math
    texts = re.findall(r"^\[(\d+)\] (.*)$", user, flags=re.M)
    n = len(texts)
    share_tox = sum("TOXWORD" in t for _, t in texts) / max(n, 1)
    lines, tokens = [], []
    for k, t in texts:
        rnd = random.Random(hashlib.md5(t.encode()).hexdigest())
        base = 2.5 if "TOXWORD" in t else -2.5
        z = base + rnd.gauss(0, 1.5) + (1.2 * (share_tox - 0.5) if n > 1 else 0)
        p = 1 / (1 + math.exp(-z))
        lab = "TOXIC" if p >= 0.5 else "NOT_TOXIC"
        lines.append(f"{k}: {lab}")
        tokens += [(("\n" if tokens else "") + k, []), (":", []),
                   (" " + lab[:3], [(" TOX", math.log(max(p, 1e-9))), (" NOT", math.log(max(1 - p, 1e-9)))]),
                   (lab[3:], [])]
    return dict(text="\n".join(lines), tokens=tokens, finish_reason="stop", served_model="mock",
                usage=dict(input=len(user) // 4 + 250, output=6 * n))


_CALLERS = {"openai": _call_openai_compatible, "deepseek": _call_openai_compatible,
            "anthropic": _call_anthropic, "gigachat": _call_gigachat, "mock": _call_mock}


# ---------------------------------------------------------------------------------------------
# Кэш
# ---------------------------------------------------------------------------------------------
class _Cache:
    def __init__(self, model_key: str):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{model_key}.jsonl"
        self.lock = threading.Lock()
        self.data: dict = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec["value"]
                    except Exception:
                        pass     # оборванная последняя строка после аварийной остановки

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


_caches: dict = {}


def cache_for(model_key: str) -> _Cache:
    if model_key not in _caches:
        _caches[model_key] = _Cache(model_key)
    return _caches[model_key]


def request_key(spec: ModelSpec, system: str, user: str, temperature: float, max_tokens: int, sample: int) -> str:
    raw = json.dumps([spec.model, system, user, float(temperature), int(max_tokens), int(sample)], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def is_cached(model_key: str, system: str, user: str, temperature: float, max_tokens: int, sample: int) -> bool:
    return cache_for(model_key).get(request_key(MODELS[model_key], system, user, temperature, max_tokens, sample)) is not None


# ---------------------------------------------------------------------------------------------
# Публичный интерфейс
# ---------------------------------------------------------------------------------------------
def ask(model_key: str, system: str, user: str, temperature: float = 0.0, max_tokens: int = 16, sample: int = 0) -> dict:
    spec = MODELS[model_key]
    key = request_key(spec, system, user, temperature, max_tokens, sample)
    cached = cache_for(model_key).get(key)
    if cached is not None:
        return cached
    if model_key in _daily_limit_hit:      # лимит уже исчерпан: не тратим время на повторы
        return dict(text="", tokens=None, usage={}, finish_reason="error", error=_daily_limit_hit[model_key])

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True,
           retry=retry_if_not_exception_type(DailyLimitError))
    def _go():
        try:
            return _CALLERS[spec.provider](spec, system, user, temperature, max_tokens)
        except Exception as e:
            if _is_daily_limit(e):
                raise DailyLimitError(f"{type(e).__name__}: {e}") from e
            raise

    try:
        out = _go()
        out["error"] = None
        if out.get("finish_reason") != "length":
            cache_for(model_key).put(key, out)
    except DailyLimitError as e:
        _daily_limit_hit[model_key] = f"DailyLimit: {e}"[:300]
        out = dict(text="", tokens=None, usage={}, finish_reason="error", error=_daily_limit_hit[model_key])
    except Exception as e:
        out = dict(text="", tokens=None, usage={}, finish_reason="error", error=f"{type(e).__name__}: {e}")
    return out


def ask_many(model_key: str, jobs: list[dict], desc: str | None = None) -> dict:
    """jobs: [{id, system, user, max_tokens, temperature?, sample?}] -> {id: ответ}."""
    workers = MODELS[model_key].workers
    results: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ask, model_key, j["system"], j["user"], j.get("temperature", 0.0),
                          j["max_tokens"], j.get("sample", 0)): j["id"] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc or model_key):
            results[futs[fut]] = fut.result()
    return results
