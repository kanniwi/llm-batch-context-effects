"""Пути, константы дизайна эксперимента и реестр моделей.

Всё, что влияет на результат (параметры отбора данных, размер батча, уровни доли токсичных соседей,
модели и их настройки), собрано здесь, чтобы читатель видел весь дизайн в одном месте.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.getenv("CTXBIAS_ROOT", Path(__file__).resolve().parents[1]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_RES = ROOT / "data" / "results"
CACHE_DIR = ROOT / "data" / "cache"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

# ---------------------------------------------------------------------------------------------
# Отбор данных (Jigsaw Unintended Bias in Toxicity Classification)
# ---------------------------------------------------------------------------------------------
DATA_SEED = 42
MIN_ANNOTATORS = 10              # меньше разметчиков: доля 0.5 может значить «1 из 2»
MIN_WORDS, MAX_WORDS = 15, 60
MAX_NON_ASCII_SHARE = 0.05
MIN_UNIQ_WORD_SHARE = 0.5
MAX_TOKENS_PER_WORD = 2.5
TOKENIZER = "o200k_base"         # только для выравнивания длины соседей, не для оценки стоимости
N_NEIGH_PER_CLASS = 400          # пул соседей: 400 токсичных + 400 нетоксичных
N_CLEAR_TARGET_PER_CLASS = 75    # явные целевые: 75 токсичных + 75 нетоксичных
N_AMBIG_TARGET = 150             # спорные целевые (доля разметчиков 0.4–0.6)

# ---------------------------------------------------------------------------------------------
# Дизайн батча
# ---------------------------------------------------------------------------------------------
K_NEIGHBORS = 8                          # соседей в батче (+1 целевой = 9 текстов)
N_TOXIC_LEVELS = (0, 2, 4, 6, 8)         # токсичных из 8 = доли 0 / 25 / 50 / 75 / 100 %
H1_POSITIONS = (1, 5, 9)                 # позиция целевого в повторе r = 0, 1, 2
LENGTH_TOLERANCES = (0.15, 0.20, 0.30)   # допуск по длине соседей относительно целевого (в токенах)
H1_SEED = 42
H1C_SEED_TARGETS = 1                     # выборка 100 целевых для проверки позиций
H1C_SEED_PLAN = 11
H1C_SEED_PLAN_EXT = 12                    # план для 200 целевых, добавленных к первым 100
H1C_LEVELS = (0, 8)
H1C_POSITIONS = tuple(range(1, 10))
H1C_N_CLEAR_PER_CLASS = 25
H1C_N_AMBIG = 50
REPEAT_SUBSET_SEED = 3
REPEAT_SUBSET_SIZE = 300                 # батчей основного набора для проверки самосогласованности

# ---------------------------------------------------------------------------------------------
# Запросы к моделям
# ---------------------------------------------------------------------------------------------
MAX_TOKENS_SINGLE = 16
MAX_TOKENS_BATCH = 120
T_WORK = 0.0          # рабочая температура
T_UNCERTAINTY = 1.0   # температура для шкалы неуверенности модели
SAMPLE_BASE = 0
SAMPLES_REPEAT = (1, 2)                  # независимые повторы при рабочей температуре
SAMPLES_UNCERTAINTY = tuple(range(101, 111))  # 10 ответов на одну целевую при T_UNCERTAINTY


@dataclass(frozen=True)
class ModelSpec:
    key: str                 # ключ в реестре и имя файла кэша
    provider: str            # gigachat | deepseek | openai | anthropic | mock
    model: str               # идентификатор модели у провайдера (входит в ключ кэша)
    api_key_env: str | None  # переменная окружения с ключом (в .env)
    workers: int = 4         # параллельных запросов
    logprobs: bool = False
    base_url: str | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {s.key: s for s in [
    # --- использованы в статье (все четыре) -------------------------------------------------------------
    ModelSpec("gigachat_lite", "gigachat", "GigaChat-2", "GIGACHAT_CREDENTIALS", workers=1,
              notes="GigaChat API; температура передаётся как max(T, 0.001) (API не принимает 0)."),
    ModelSpec("gigachat_ultra", "gigachat", "GigaChat-3-Ultra", "GIGACHAT_CREDENTIALS", workers=1,
              notes="GigaChat 3 Ultra (бесплатный пакет 50 млн токенов); температура передаётся как max(T, 0.001)."),
    ModelSpec("deepseek", "deepseek", "deepseek-flash", "DEEPSEEK_API_KEY", workers=8,
              base_url="https://api.deepseek.com",
              notes="Официальный API DeepSeek; режим размышления отключается thinking={'type':'disabled'}."),
    ModelSpec("openai", "openai", "gpt-5.6-terra", "OPENAI_API_KEY", workers=8, logprobs=True,
              notes="reasoning_effort=none; logprobs отключаются сами, если не поддержаны."),
    ModelSpec("claude", "anthropic", "claude-sonnet-5-5", "ANTHROPIC_API_KEY", workers=4,
              notes="Anthropic Messages API; температура 0 не принимается API (запросы идут при значении по умолчанию); logprobs недоступны."),
    # --- только для тестов ------------------------------------------------------------------
    ModelSpec("mock", "mock", "mock", None, workers=8, logprobs=True, notes="Имитация модели для автотестов."),
]}
