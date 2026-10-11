MODELS = {
    "deepseek": dict(provider="deepseek", model="deepseek-flash", logprobs=True, workers=8),
    "openai": dict(provider="openai", model="gpt-5.6-terra", logprobs=True, workers=8),   # если logprobs не поддерживаются само отключится
    "claude": dict(provider="anthropic", model="claude-sonnet-5-5", logprobs=False, workers=4),
    "gigachat": dict(provider="gigachat", model="GigaChat-2-Max", logprobs=False, workers=1),  
    "gigachat_lite": dict(provider="gigachat", model="GigaChat-2", logprobs=False, workers=1), 
    "groq": dict(provider="groq", model="openai/gpt-oss-120b", logprobs=False, workers=2),
    "groq20": dict(provider="groq", model="openai/gpt-oss-20b", logprobs=False, workers=2),
    "mock": dict(provider="mock", model="mock", logprobs=True, workers=8),               
}

ENV_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gigachat": "GIGACHAT_CREDENTIALS",
    "groq": "GROQ_API_KEY",
}
