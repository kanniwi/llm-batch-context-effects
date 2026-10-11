import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

from .config import ENV_KEYS, MODELS

PROJECT_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_DIR / "data" / "cache"
load_dotenv(PROJECT_DIR / ".env", override=True)

_clients = {}
_logprobs_supported = {}
_daily_limit_hit = {}   # model_key -> текст ошибки, если исчерпан ДНЕВНОЙ лимит провайдера


class DailyLimitError(Exception):
    """Дневной лимит (запросов или токенов) исчерпан — повторять сегодня бесполезно."""


def _is_daily_limit(e) -> bool:
    msg = str(e).lower()
    return "per day" in msg or "(tpd)" in msg or "(rpd)" in msg


def daily_limit_hit(model_key: str):
    """Текст ошибки, если дневной лимит модели исчерпан в этой сессии, иначе None."""
    return _daily_limit_hit.get(model_key)


def reset_daily_limit(model_key: str = None):
    """Сбросить флаг (например, на следующий день, не перезапуская ядро)."""
    if model_key is None:
        _daily_limit_hit.clear()
    else:
        _daily_limit_hit.pop(model_key, None)


def _client(provider):
    if provider in _clients:
        return _clients[provider]
    if provider in ENV_KEYS and not os.getenv(ENV_KEYS[provider]):
        raise RuntimeError(f"Нет ключа {ENV_KEYS[provider]} в файле .env")
    if provider == "openai":
        from openai import OpenAI
        c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "deepseek":
        from openai import OpenAI
        c = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    elif provider == "groq":
        from openai import OpenAI
        c = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    elif provider == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "gigachat":
        from gigachat import GigaChat
        c = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            verify_ssl_certs=False,  
            profanity_check=False,  
            timeout=120,
        )
    else:
        c = None
    _clients[provider] = c
    return c


def _call_openai_compatible(cfg, system, user, temperature, max_tokens):
    client = _client(cfg["provider"])
    kwargs = dict(model=cfg["model"], temperature=temperature,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if cfg["provider"] == "openai":
        kwargs["max_completion_tokens"] = max_tokens
        kwargs["reasoning_effort"] = "none"  
    elif cfg["provider"] == "groq":
        kwargs["max_completion_tokens"] = max_tokens
        if "gpt-oss" in cfg["model"]:
            kwargs["reasoning_effort"] = "low"
            kwargs["max_completion_tokens"] = max_tokens + 800
            kwargs["extra_body"] = {"include_reasoning": False}
    else:
        kwargs["max_tokens"] = max_tokens
    want_lp = cfg["logprobs"] and _logprobs_supported.get(cfg["model"], True)
    if want_lp:
        kwargs.update(logprobs=True, top_logprobs=10)
    try:
        r = client.chat.completions.create(**kwargs)
    except Exception as e: 
        if want_lp and "logprob" in str(e).lower():
            _logprobs_supported[cfg["model"]] = False
            kwargs.pop("logprobs"), kwargs.pop("top_logprobs")
            r = client.chat.completions.create(**kwargs)
        else:
            raise
    ch = r.choices[0]
    tokens = None
    if getattr(ch, "logprobs", None) and ch.logprobs.content:
        tokens = [(t.token, [(a.token, a.logprob) for a in (t.top_logprobs or [])]) for t in ch.logprobs.content]
    usage = dict(input=r.usage.prompt_tokens, output=r.usage.completion_tokens) if r.usage else {}
    return dict(text=ch.message.content or "", tokens=tokens, usage=usage, finish_reason=ch.finish_reason)


def _call_anthropic(cfg, system, user, temperature, max_tokens):
    r = _client("anthropic").messages.create(
        model=cfg["model"], system=system, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return dict(text=text, tokens=None, finish_reason=r.stop_reason,
                usage=dict(input=r.usage.input_tokens, output=r.usage.output_tokens))


def _call_gigachat(cfg, system, user, temperature, max_tokens):
    from gigachat.models import Chat, Messages, MessagesRole
    chat = Chat(model=cfg["model"], max_tokens=max_tokens, profanity_check=False,
                temperature=max(temperature, 0.001),  
                messages=[Messages(role=MessagesRole.SYSTEM, content=system),
                          Messages(role=MessagesRole.USER, content=user)])
    r = _client("gigachat").chat(chat)
    ch = r.choices[0]
    return dict(text=ch.message.content or "", tokens=None, finish_reason=ch.finish_reason, 
                usage=dict(input=r.usage.prompt_tokens, output=r.usage.completion_tokens))


def _call_mock(cfg, system, user, temperature, max_tokens):
    import math, re
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
    return dict(text="\n".join(lines), tokens=tokens, finish_reason="stop",
                usage=dict(input=len(user) // 4 + 250, output=6 * n))


_CALLERS = {"openai": _call_openai_compatible, "deepseek": _call_openai_compatible, "groq": _call_openai_compatible,
            "anthropic": _call_anthropic, "gigachat": _call_gigachat, "mock": _call_mock}


class _Cache:
    def __init__(self, model_key):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{model_key}.jsonl"
        self.lock = threading.Lock()
        self.data = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec["value"]
                    except Exception:
                        pass 

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


_caches = {}


def _cache(model_key):
    if model_key not in _caches:
        _caches[model_key] = _Cache(model_key)
    return _caches[model_key]


def _key(cfg, system, user, temperature, max_tokens, sample):
    raw = json.dumps([cfg["model"], system, user, temperature, max_tokens, sample], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()



def ask(model_key, system, user, temperature=0.0, max_tokens=None, sample=0):
    cfg = MODELS[model_key]
    max_tokens = max_tokens or 16
    key = _key(cfg, system, user, temperature, max_tokens, sample)
    cached = _cache(model_key).get(key)
    if cached is not None:
        return cached

    if model_key in _daily_limit_hit:   # лимит уже исчерпан — не тратим время на повторы
        return dict(text="", tokens=None, usage={}, finish_reason="error", error=_daily_limit_hit[model_key])

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True,
           retry=retry_if_not_exception_type(DailyLimitError))
    def _go():
        try:
            return _CALLERS[cfg["provider"]](cfg, system, user, temperature, max_tokens)
        except Exception as e:
            if _is_daily_limit(e):
                raise DailyLimitError(f"{type(e).__name__}: {e}") from e
            raise

    try:
        out = _go()
        out["error"] = None
        if out.get("finish_reason") != "length": 
            _cache(model_key).put(key, out) 
    except DailyLimitError as e:
        _daily_limit_hit[model_key] = f"DailyLimit: {e}"[:300]
        out = dict(text="", tokens=None, usage={}, finish_reason="error", error=_daily_limit_hit[model_key])
    except Exception as e:
        out = dict(text="", tokens=None, usage={}, finish_reason="error", error=f"{type(e).__name__}: {e}")
    return out


def ask_many(model_key, jobs, desc=None):
    workers = MODELS[model_key]["workers"]
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ask, model_key, j["system"], j["user"], j.get("temperature", 0.0),
                          j["max_tokens"], j.get("sample", 0)): j["id"] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc or model_key):
            results[futs[fut]] = fut.result()
    return results
