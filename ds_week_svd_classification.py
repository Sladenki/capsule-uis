# -*- coding: utf-8 -*-
"""
Спектральные методы снижения размерности (SVD) и модели классификации
в задаче автоматического синтеза UI-интерфейсов из текста
(на примере платформы CapsuleOS).

Демонстрационный скрипт для постерной конференции Data Science Week.

Пайплайн
--------
1. Синтетический датасет из 500 промптов и мультиметок UI-компонентов.
2. Очистка текста, токенизация, нормализация словоформ, TF-IDF (p <= 1000).
3. TruncatedSVD до k = 30 и спектр накопленной энергии F_k для k = 1..50.
4. Обучение LogisticRegression и SVC (RBF) на латентных признаках.
5. Сравнение Accuracy и ROC-AUC, графики и отчёт с цифрами для постера.

Векторизатор, SVD и стандартизация обучаются только на train (70%),
тестовая выборка (30%) лишь проецируется в уже построенное пространство.
Так в метрики не попадает утечка idf-статистик и сингулярного базиса.

Зависимости: pandas, numpy, matplotlib, scikit-learn.
Запуск: python ds_week_svd_classification.py
"""

from __future__ import annotations

import re
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # сохранение рисунков без оконного интерфейса

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Параметры эксперимента (их же переносим на постер)
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_PROMPTS = 500
TEST_SIZE = 0.30
TFIDF_MAX_FEATURES = 1000  # верхняя граница словаря p
SVD_COMPONENTS = 30        # рабочая латентная размерность k
SVD_SPECTRUM = 50          # до какой компоненты строится кривая F_k
N_INFERENCE_REPEATS = 30

LABELS = ["Form", "Table", "Kanban", "PhotoUploader"]
LABELS_RU = {
    "Form": "Форма",
    "Table": "Таблица",
    "Kanban": "Канбан",
    "PhotoUploader": "Загрузка фото",
}

BASE_DIR = Path(__file__).resolve().parent
PATH_VARIANCE = BASE_DIR / "svd_explained_variance.png"
PATH_COMPARISON = BASE_DIR / "models_comparison.png"
PATH_REPORT = BASE_DIR / "poster_metrics.txt"
PATH_DATASET = BASE_DIR / "synthetic_prompts.csv"

# Цвета, различимые и на цветной печати постера, и при ч/б-копии (штриховка).
COLOR_ENERGY = "#0072B2"
COLOR_MARK = "#D55E00"
COLOR_ACCURACY = "#0072B2"
COLOR_AUC = "#E69F00"


# ---------------------------------------------------------------------------
# Нормализация словоформ
# ---------------------------------------------------------------------------
# Полноценный морфологический анализатор (pymorphy) сознательно не подключаем:
# скрипт должен запускаться на стеке pandas / numpy / matplotlib / scikit-learn.
# Ниже — прозрачная лемматизация учебных промптов:
#   1) словарь исключений для нерегулярных парадигм (снимок, колонок, ...);
#   2) однократное отсечение регулярного окончания при основе не короче 3 букв.
# Этого достаточно, чтобы «таблицу» и «таблицей» попали в один токен TF-IDF.

VOWELS = set("аеёиоуыэюя")

# Длинные окончания раньше коротких: «ами» должно сработать раньше «и».
_RAW_SUFFIXES = [
    "ениями", "ениях", "ением", "ению", "ении", "ения", "ение",
    "иями", "ями", "ами", "иях", "ого", "его", "ему", "ому", "ыми", "ими",
    "остями", "остью", "ости", "ость", "ями", "иях", "ах", "ях", "ая", "яя", "ое", "ее",
    "ые", "ие", "ую", "юю", "ою", "ею", "ый", "ий", "ой", "ей", "ых", "их",
    "ия", "ии", "ию", "ья", "ью", "ев", "ов", "ам", "ям", "ом", "ем",
    "а", "я", "ы", "и", "е", "о", "у", "ю", "й", "ь",
]
SUFFIXES = tuple(sorted(set(_RAW_SUFFIXES), key=len, reverse=True))

# Нерегулярные формы, которые суффиксный шаг не сводит к одной лемме.
LEMMA_EXCEPTIONS = {
    "снимок": "снимок",
    "снимка": "снимок",
    "снимку": "снимок",
    "снимком": "снимок",
    "снимке": "снимок",
    "снимки": "снимок",
    "снимков": "снимок",
    "колонок": "колонк",
    "досок": "доск",
    "заявок": "заявк",
    "анкет": "анкет",
    "форм": "форм",
}

# Служебные слова и глаголы-просьбы: они есть почти в каждом промпте
# и не отличают один UI-компонент от другого.
STOP_WORDS = {
    "и", "в", "во", "на", "с", "со", "по", "для", "из", "к", "ко", "от", "до",
    "за", "о", "об", "а", "но", "или", "что", "это", "как", "так", "у", "не",
    "да", "же", "бы", "ли", "то", "все", "всё", "его", "ее", "её", "их", "мы",
    "вы", "они", "она", "он", "мне", "меня", "мой", "моя", "мои", "ваш", "наш",
    "сделай", "сделать", "добавь", "добавить", "покажи", "показать", "собери",
    "собрать", "сверстай", "выведи", "построй", "размести", "отрисуй",
    "спроектируй", "нужен", "нужна", "нужно", "нужны", "хочу", "видеть",
    "экран", "экране", "экрана", "интерфейс", "интерфейса", "страница",
    "страницу", "страницы", "раздел", "раздела", "модуль", "модуле", "модуля",
    "capsuleos", "пожалуйста", "просто", "очень", "ещё", "еще", "уже", "только",
    "этот", "эта", "эти", "этого", "этой", "весь", "вся", "всю", "всего",
    "пользователь", "пользователя", "данные", "данных", "данными",
}


def lemmatize_token(token: str) -> str:
    """Приводит словоформу к учебной лемме (исключение или срез окончания)."""
    token = token.replace("ё", "е")
    if token in LEMMA_EXCEPTIONS:
        return LEMMA_EXCEPTIONS[token]
    # Латиница (workflow, capsuleos) остаётся как есть: окончаний русского языка там нет.
    if not any("а" <= ch <= "я" for ch in token):
        return token
    for suffix in SUFFIXES:
        if len(token) - len(suffix) < 3 or not token.endswith(suffix):
            continue
        stem = token[: -len(suffix)]
        if any(ch in VOWELS for ch in stem):
            return stem
    return token


def _check_paradigms() -> None:
    """Страхует лемматизатор: формы одного слова обязаны схлопываться."""
    paradigms = [
        ["форма", "форму", "формы", "формой", "форме"],
        ["таблица", "таблицу", "таблицы", "таблицей"],
        ["канбан", "канбана", "канбаном", "канбане"],
        ["доска", "доску", "доски", "доской", "доске", "досок"],
        ["задача", "задачи", "задачу", "задачей"],
        ["поле", "поля", "полем", "полей"],
        ["ввод", "ввода", "вводу", "вводом"],
        ["загрузка", "загрузку", "загрузки", "загрузкой"],
        ["аналитика", "аналитику", "аналитикой"],
        ["реестр", "реестра", "реестры"],
        ["ведомость", "ведомости", "ведомостью"],
        ["бэклог", "бэклога"],
        ["спринт", "спринта", "спринтов"],
        ["отчёт", "отчета", "отчёты"],
        ["снимок", "снимка", "снимку", "снимки", "снимков"],
        ["колонка", "колонки", "колонку", "колонками", "колонок"],
        ["заявка", "заявку", "заявки", "заявок"],
    ]
    for forms in paradigms:
        lemmas = {lemmatize_token(form) for form in forms}
        if len(lemmas) != 1:
            raise RuntimeError(f"Парадигма не сошлась: {forms} -> {lemmas}")


_TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)


def preprocess(text: str) -> list[str]:
    """Очистка, нижний регистр, токены, стоп-слова, лемматизация.

    Если после стоп-слов текст опустел, возвращаем леммы без фильтра,
    чтобы у промпта не получилась нулевая строка TF-IDF.
    """
    lowered = text.lower().replace("ё", "е")
    tokens = _TOKEN_RE.findall(lowered)
    content = []
    fallback = []
    for token in tokens:
        if len(token) < 2:
            continue
        lemma = lemmatize_token(token)
        if len(lemma) < 2:
            continue
        fallback.append(lemma)
        if token in STOP_WORDS or lemma in STOP_WORDS:
            continue
        content.append(lemma)
    return content or fallback


# ---------------------------------------------------------------------------
# Генерация синтетических промптов
# ---------------------------------------------------------------------------
# Метки не случайны: фраза компонента добавляется только если метка равна 1.
# Слоты «домен / роль / метрика / отрасль» общие для всех классов, поэтому
# различительный сигнал сидит в лексемах компонента (форма, реестр, канбан, фото),
# а не в теме бизнеса. Так SVD есть что сжимать: тематический шум отделяется
# от низкорангового сигнала о составе интерфейса.

DOMAINS = [
    "курьера", "заказов", "клиентов", "склада", "маржи", "проекта",
    "сотрудников", "поставок", "обращений", "платежей", "вакансий",
    "инвентаризации", "договоров", "смен", "маршрутов", "отзывов",
    "товаров", "услуг", "лидов", "бронирований", "накладных", "рейсов",
    "пациентов", "студентов", "оборудования", "бюджета", "рисков",
    "инцидентов", "документов", "кандидатов", "поставщиков", "счетов",
    "остатков", "филиалов", "контрактов", "возвратов", "отгрузок",
    "партий", "чеков", "водителей", "операторов", "менеджеров",
    "кассиров", "бухгалтерии", "логистики", "доставки", "поддержки",
    "продаж", "закупки", "производства", "качества", "охраны",
    "обучения", "найма", "визитов", "осмотров", "образцов", "тикетов",
    "релизов", "гипотез", "кампаний", "креативов", "баннеров",
    "тарифов", "начислений", "выплат", "штрафов", "премий", "графиков",
    "табеля", "бригад", "площадок", "объектов", "серий", "квот",
    "касс", "проб", "фактов", "планов",
]

ROLES = [
    "курьера", "менеджера", "оператора", "кладовщика", "врача",
    "водителя", "диспетчера", "агента", "инспектора", "преподавателя",
    "кассира", "мастера", "технолога", "бухгалтера", "рекрутера",
    "логиста", "закупщика", "супервайзера", "координатора", "приёмщика",
]

METRICS = [
    "маржи", "скидки", "ндс", "зарплаты", "себестоимости", "конверсии",
    "комиссии", "рентабельности", "утилизации", "оборачиваемости",
    "выручки", "прибыли", "налога", "премии", "стоимости", "веса",
    "срока",     "расстояния", "окупаемости", "брака", "текучести", "ltv",
]

INDUSTRIES = [
    "ритейла", "логистики", "медицины", "образования", "финтеха",
    "производства", "horeca", "агросектора", "телекома", "строительства",
    "страхования", "маркетинга", "юридического контура", "hr",
]

OPENERS = [
    "сделай", "добавь", "покажи", "собери", "сверстай",
    "выведи", "построй", "размести", "отрисуй", "спроектируй",
]

# Фразы в именительном падеже (режим заголовка) и в винительном (после «сделай»).
PHRASES = {
    "Form": {
        "nom": [
            "форма {domain}",
            "анкета {domain}",
            "бланк регистрации {domain}",
            "калькулятор {metric} с полем ввода",
            "поля ввода для {domain}",
            "опросник {domain}",
            "форма заявки {role}",
            "бланк с полями ввода для {role}",
        ],
        "acc": [
            "форму {domain}",
            "анкету {domain}",
            "бланк регистрации {domain}",
            "калькулятор {metric} с полем ввода",
            "поля ввода для {domain}",
            "опросник {domain}",
            "форму заявки {role}",
            "бланк с полями ввода для {role}",
        ],
    },
    "Table": {
        "nom": [
            "таблица {domain}",
            "реестр {domain}",
            "аналитическая сводка для {domain}",
            "журнал записей {domain}",
            "ведомость {domain}",
            "таблица с колонками для {domain}",
            "отчёт в виде таблицы {domain}",
            "аналитика {domain}",
        ],
        "acc": [
            "таблицу {domain}",
            "реестр {domain}",
            "аналитическую сводку для {domain}",
            "журнал записей {domain}",
            "ведомость {domain}",
            "таблицу с колонками для {domain}",
            "отчёт в виде таблицы {domain}",
            "аналитику {domain}",
        ],
    },
    "Kanban": {
        "nom": [
            "канбан {domain}",
            "канбан-доска {domain}",
            "доска задач {domain}",
            "канбан проектных задач",
            "бэклог с колонками статусов {domain}",
            "доска спринтов {domain}",
            "карточки задач для {role}",
            "workflow-доска {domain}",
        ],
        "acc": [
            "канбан {domain}",
            "канбан-доску {domain}",
            "доску задач {domain}",
            "канбан проектных задач",
            "бэклог с колонками статусов {domain}",
            "доску спринтов {domain}",
            "карточки задач для {role}",
            "workflow-доску {domain}",
        ],
    },
    "PhotoUploader": {
        "nom": [
            "загрузка фото {domain}",
            "поле фотографии {role}",
            "блок фотофиксации {domain}",
            "прикрепление снимка",
            "камера с превью снимка",
            "загрузчик изображений",
            "слот фотографии с превью",
            "фотофиксация для {role}",
        ],
        "acc": [
            "загрузку фото {domain}",
            "поле фотографии {role}",
            "блок фотофиксации {domain}",
            "прикрепление снимка",
            "камеру с превью снимка",
            "загрузчик изображений",
            "слот фотографии с превью",
            "фотофиксацию для {role}",
        ],
    },
}

MODIFIERS = [
    "в модуле capsuleos",
    "для мобильной версии",
    "для десктопной версии",
    "в тёмной теме",
    "в светлой теме",
    "на главном экране",
    "во весь экран",
    "с крупной типографикой",
    "для внутренней команды",
    "с подсказками интерфейса",
    "с компактными отступами",
    "для ночной смены",
    "с быстрыми действиями",
    "без лишнего декора",
]

# Канонические формулировки из постановки задачи — они входят в выборку как есть.
CANONICAL = [
    {
        "prompt": "сделай форму курьера с загрузкой фото",
        "Form": 1, "Table": 0, "Kanban": 0, "PhotoUploader": 1,
    },
    {
        "prompt": "таблица заказов и аналитика",
        "Form": 0, "Table": 1, "Kanban": 0, "PhotoUploader": 0,
    },
    {
        "prompt": "канбан проектных задач",
        "Form": 0, "Table": 0, "Kanban": 1, "PhotoUploader": 0,
    },
    {
        "prompt": "калькулятор маржи с полем ввода",
        "Form": 1, "Table": 0, "Kanban": 0, "PhotoUploader": 0,
    },
]


def _join_ru(parts: list[str]) -> str:
    """Соединяет фрагменты по-русски: «а, б и в»."""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} и {parts[1]}"
    return ", ".join(parts[:-1]) + " и " + parts[-1]


def _fill(template: str, rng: np.random.Generator) -> str:
    return template.format(
        domain=rng.choice(DOMAINS),
        role=rng.choice(ROLES),
        metric=rng.choice(METRICS),
        industry=rng.choice(INDUSTRIES),
    )


def _render_prompt(active: tuple[str, ...], rng: np.random.Generator) -> str:
    """Собирает грамматически цельный промпт под заданный набор меток."""
    accusative = bool(rng.random() < 0.65)
    mode = "acc" if accusative else "nom"
    chunks = [_fill(rng.choice(PHRASES[label][mode]), rng) for label in active]
    rng.shuffle(chunks)
    body = _join_ru(chunks)
    opener = f"{rng.choice(OPENERS)} " if accusative else ""
    industry = f" в контуре {rng.choice(INDUSTRIES)}" if rng.random() < 0.35 else ""
    modifier = f" {rng.choice(MODIFIERS)}" if rng.random() < 0.55 else ""
    text = f"{opener}{body}{industry}{modifier}"
    return re.sub(r"\s+", " ", text).strip()


def generate_dataset(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Равномерно покрывает все 15 непустых комбинаций четырёх компонентов.

    Равномерность по комбинациям нужна, чтобы каждая метка встречалась
    и в train, и в test: иначе ROC-AUC для редкого компонента не определён.
    """
    subsets = [
        combo
        for size in range(1, len(LABELS) + 1)
        for combo in combinations(LABELS, size)
    ]
    plan = [subsets[i % len(subsets)] for i in range(n)]
    rng.shuffle(plan)

    rows = []
    seen: set[str] = set()
    for active in plan:
        prompt = _render_prompt(active, rng)
        # Повторно собираем фразу, если такая строка уже есть: одинаковый текст
        # в train и test завышал бы качество.
        for _ in range(40):
            if prompt not in seen:
                break
            prompt = _render_prompt(active, rng)
        seen.add(prompt)
        row = {label: int(label in active) for label in LABELS}
        row["prompt"] = prompt
        rows.append(row)

    frame = pd.DataFrame(rows)
    # Канонические фразы подменяют случайные строки с теми же метками,
    # чтобы в датасете дословно были примеры из постановки.
    for sample in CANONICAL:
        mask = np.ones(len(frame), dtype=bool)
        for label in LABELS:
            mask &= frame[label].to_numpy() == sample[label]
        indexes = np.flatnonzero(mask)
        if len(indexes) == 0:
            continue
        frame.loc[indexes[0], "prompt"] = sample["prompt"]

    # Первое вхождение каждой фразы оставляем (в том числе канон), копии пересобираем.
    seen = set(frame["prompt"])
    duplicates = frame["prompt"].duplicated(keep="first")
    for position in np.flatnonzero(duplicates.to_numpy()):
        active = tuple(label for label in LABELS if frame.at[position, label] == 1)
        prompt = _render_prompt(active, rng)
        for _ in range(40):
            if prompt not in seen:
                break
            prompt = _render_prompt(active, rng)
        seen.add(prompt)
        frame.at[position, "prompt"] = prompt

    frame = frame[["prompt", *LABELS]]
    return frame.reset_index(drop=True)


def _content_lemmas(text: str) -> set[str]:
    return set(preprocess(text))


def _audit_signal_leak() -> None:
    """Проверяет, что общие модификаторы не содержат лексем UI-компонентов.

    Доменные слова специально не проверяем: они подставляются во все классы
    и не являются чистым сигналом метки.
    """
    cue_bank = []
    for label in LABELS:
        for mode in ("nom", "acc"):
            for template in PHRASES[label][mode]:
                bare = template.format(domain="", role="", metric="", industry="")
                cue_bank.extend(preprocess(bare))
    cues = set(cue_bank)

    shared_chunks = MODIFIERS + INDUSTRIES + OPENERS + DOMAINS + ROLES
    leaks = []
    for chunk in shared_chunks:
        hit = _content_lemmas(chunk) & cues
        if hit:
            leaks.append(f"{chunk!r}: {sorted(hit)}")
    if leaks:
        joined = "\n".join(leaks)
        raise RuntimeError(
            "Общие слоты содержат лексемы UI-компонентов, сигнал смешается:\n" + joined
        )


# ---------------------------------------------------------------------------
# Модели и метрики
# ---------------------------------------------------------------------------

def _positive_scores(model: OneVsRestClassifier, features: np.ndarray) -> np.ndarray:
    """Вероятности положительного класса по каждой из 4 меток, форма (n, 4)."""
    proba = model.predict_proba(features)
    if isinstance(proba, list):
        return np.column_stack([column[:, 1] for column in proba])
    return np.asarray(proba)


def _label_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Доля верных бинарных решений, усреднённая по объектам и меткам.

    Совпадает с 1 − Hamming loss. Для мультилейбла это естественный Accuracy:
    каждый UI-компонент — отдельное решение «рисовать / не рисовать».
    """
    return float(1.0 - hamming_loss(y_true, y_pred))


def _per_label_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.array([
        accuracy_score(y_true[:, index], y_pred[:, index]) for index in range(y_true.shape[1])
    ])


def _measure_predict_ms(model: OneVsRestClassifier, features: np.ndarray, repeats: int) -> float:
    """Среднее время model.predict на тестовом батче, миллисекунды."""
    model.predict(features)  # прогрев, чтобы в замер не попала ленивая инициализация
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict(features)
    elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / repeats


def _top_tfidf_tokens(
    matrix: np.ndarray,
    y_train: np.ndarray,
    feature_names: np.ndarray,
    top_n: int = 6,
) -> dict[str, list[str]]:
    """Стемы, у которых средний TF-IDF в положительном классе заметно выше."""
    result = {}
    for index, label in enumerate(LABELS):
        positive = y_train[:, index] == 1
        negative = ~positive
        contrast = np.asarray(matrix[positive].mean(axis=0)).ravel()
        contrast -= np.asarray(matrix[negative].mean(axis=0)).ravel()
        best = np.argsort(contrast)[-top_n:][::-1]
        result[label] = [str(feature_names[item]) for item in best]
    return result


def _top_svd_terms(svd: TruncatedSVD, feature_names: np.ndarray, n_components: int = 5, top_n: int = 6) -> list[str]:
    """Самые тяжёлые токены первых латентных компонент (знак компоненты условен)."""
    lines = []
    for index in range(n_components):
        weights = svd.components_[index]
        best = np.argsort(np.abs(weights))[-top_n:][::-1]
        terms = [f"{feature_names[item]} ({weights[item]:+.2f})" for item in best]
        lines.append(f"  компонента {index + 1}: " + ", ".join(terms))
    return lines


# ---------------------------------------------------------------------------
# Рисунки для постера
# ---------------------------------------------------------------------------

def _apply_poster_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_explained_variance(cumulative: np.ndarray, energy_at_k: float, path: Path) -> None:
    """Кривая F_k и отметка рабочей размерности k = 30."""
    ks = np.arange(1, len(cumulative) + 1)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(
        ks,
        cumulative,
        color=COLOR_ENERGY,
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        label=r"Накопленная энергия $F_k$",
    )
    ax.fill_between(ks, cumulative, color=COLOR_ENERGY, alpha=0.12)
    ax.axvline(SVD_COMPONENTS, color=COLOR_MARK, linestyle="--", linewidth=1.4)
    ax.scatter([SVD_COMPONENTS], [energy_at_k], color=COLOR_MARK, s=55, zorder=5)
    ax.annotate(
        f"k = {SVD_COMPONENTS}\n$F_{{{SVD_COMPONENTS}}}$ = {energy_at_k:.3f}"
        f"\n({energy_at_k * 100:.1f}% энергии)",
        xy=(SVD_COMPONENTS, energy_at_k),
        xytext=(SVD_COMPONENTS + 6, max(0.12, energy_at_k - 0.22)),
        fontsize=11,
        color=COLOR_MARK,
        arrowprops={"arrowstyle": "->", "color": COLOR_MARK, "lw": 1.2},
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COLOR_MARK, "lw": 1.0},
    )
    ax.set_xlim(1, len(cumulative))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Число латентных компонент k")
    ax.set_ylabel(r"Накопленная объяснённая дисперсия $F_k$")
    ax.set_title("Спектральное сжатие TF-IDF методом TruncatedSVD")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#DDDDDD")
    fig.text(
        0.01,
        0.01,
        "TruncatedSVD по матрице TF-IDF (LSA), без центрирования разреженных данных. "
        "Спектр оценён на обучающей выборке.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> None:
    """Групповая столбчатая диаграмма Accuracy и ROC-AUC двух моделей."""
    names = summary["Модель"].tolist()
    accuracy = summary["Accuracy"].to_numpy()
    roc_auc = summary["ROC-AUC"].to_numpy()
    x = np.arange(len(names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bars_acc = ax.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color=COLOR_ACCURACY,
        edgecolor="black",
        linewidth=0.6,
        hatch="//",
    )
    bars_auc = ax.bar(
        x + width / 2,
        roc_auc,
        width,
        label="ROC-AUC",
        color=COLOR_AUC,
        edgecolor="black",
        linewidth=0.6,
        hatch="..",
    )
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0, label="Случайный уровень 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Значение метрики")
    ax.set_xlabel("Модель")
    ax.set_title("Классификация UI-компонентов CapsuleOS в пространстве SVD")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    # Легенда вынесена вправо: столбцы почти до 1.0, внутри осей она перекрывала бы SVC.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=False,
        borderaxespad=0.0,
    )
    for bars in (bars_acc, bars_auc):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )
    fig.text(
        0.01,
        0.01,
        "Accuracy — доля верных решений по 4 компонентам (1 − Hamming loss). "
        "ROC-AUC — macro average по тем же меткам. Тест 30%, random_state = 42.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 0.78, 1))
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_report(
    n_train: int,
    n_test: int,
    n_features: int,
    cumulative: np.ndarray,
    summary: pd.DataFrame,
    per_label: pd.DataFrame,
    balance: pd.Series,
    examples: pd.DataFrame,
    token_lines: list[str],
    svd_lines: list[str],
    versions: str,
) -> str:
    energy = float(cumulative[SVD_COMPONENTS - 1])
    spectrum_rows = []
    for k in list(range(5, len(cumulative) + 1, 5)) + [SVD_COMPONENTS]:
        k = int(k)
        if 1 <= k <= len(cumulative):
            spectrum_rows.append(f"  k = {k:2d}    F_k = {cumulative[k - 1]:.4f}    ({cumulative[k - 1] * 100:.1f}%)")
    # k = 30 уже может совпасть с шагом 5; дубли убираем, порядок сохраняем.
    spectrum_rows = list(dict.fromkeys(spectrum_rows))

    poster_csv = ["Модель;Accuracy;SubsetAccuracy;ROC-AUC;p;k;F_k;ВремяИнференса_мс;мс_на_объект"]
    for _, row in summary.iterrows():
        poster_csv.append(
            ";".join([
                str(row["Модель"]),
                _fmt(row["Accuracy"]),
                _fmt(row["SubsetAccuracy"]),
                _fmt(row["ROC-AUC"]),
                str(n_features),
                str(SVD_COMPONENTS),
                _fmt(energy),
                f"{row['Время_мс']:.2f}",
                f"{row['мс_на_объект']:.4f}",
            ])
        )

    per_label_csv = ["Модель;Компонент;Accuracy;ROC-AUC"]
    for _, row in per_label.iterrows():
        per_label_csv.append(
            f"{row['Модель']};{row['Компонент']};{_fmt(row['Accuracy'])};{_fmt(row['ROC-AUC'])}"
        )

    best = summary.sort_values("ROC-AUC", ascending=False).iloc[0]
    sentence = (
        f"Матрица TF-IDF размерности p = {n_features} сжата TruncatedSVD до k = {SVD_COMPONENTS} "
        f"с сохранением {energy * 100:.1f}% спектральной энергии (F_{SVD_COMPONENTS} = {energy:.3f}). "
        f"На тесте (n = {n_test}) Accuracy / ROC-AUC: "
        + ", ".join(
            f"{row['Модель']} {_fmt(row['Accuracy'])} / {_fmt(row['ROC-AUC'])}"
            for _, row in summary.iterrows()
        )
        + f". Выше ROC-AUC у модели {best['Модель']}."
    )

    example_lines = []
    for _, row in examples.iterrows():
        active = [LABELS_RU[label] for label in LABELS if row[label] == 1]
        example_lines.append(f"  • {row['prompt']}")
        example_lines.append(f"    метки: {', '.join(active)}")

    balance_lines = [
        f"  {LABELS_RU[label]} ({label}): {int(balance[label])} из {N_PROMPTS} "
        f"({balance[label] / N_PROMPTS * 100:.1f}%)"
        for label in LABELS
    ]

    parts = [
        "Спектральные методы снижения размерности (SVD) и классификация UI-компонентов",
        "Платформа: CapsuleOS",
        "Постерная конференция Data Science Week",
        "",
        "ЦИФРЫ ДЛЯ ПОСТЕРА (можно переносить в таблицу как есть)",
        "-------------------------------------------------------",
        f"N                                 = {N_PROMPTS}",
        f"Train / test                      = {n_train} / {n_test}  (70/30, random_state = {RANDOM_STATE})",
        f"Словарь TF-IDF, ограничение p     = {TFIDF_MAX_FEATURES}",
        f"Фактическое число признаков p     = {n_features}",
        f"Латентная размерность k           = {SVD_COMPONENTS}",
        f"Накопленная энергия F_30          = {energy:.4f}  ({energy * 100:.1f}%)",
        "",
        "Модель | Accuracy | Subset Accuracy | ROC-AUC | k | t инференса, мс | мс/объект",
    ]
    for _, row in summary.iterrows():
        parts.append(
            f"{row['Модель']} | {_fmt(row['Accuracy'])} | {_fmt(row['SubsetAccuracy'])} | "
            f"{_fmt(row['ROC-AUC'])} | {SVD_COMPONENTS} | {row['Время_мс']:.2f} | {row['мс_на_объект']:.4f}"
        )
    parts.extend([
        "",
        "Блок для вставки в Excel (разделитель «;»):",
        *poster_csv,
        "",
        "ФОРМУЛИРОВКА ДЛЯ ПОДПИСИ НА ПОСТЕРЕ",
        "-----------------------------------",
        sentence,
        "",
        "ОПРЕДЕЛЕНИЯ МЕТРИК",
        "------------------",
        "Accuracy — доля верно принятых бинарных решений по 4 компонентам",
        "(Form, Table, Kanban, PhotoUploader). Совпадает с 1 − Hamming loss.",
        "Subset Accuracy — доля промптов, у которых верно предсказан весь набор",
        "компонентов сразу. Метрика жёстче и на постере идёт отдельной колонкой.",
        "ROC-AUC — macro average четырёх бинарных AUC по вероятностям predict_proba.",
        "Для SVC вероятности получены калибровкой Платта (probability=True).",
        "Время инференса — среднее model.predict на всём тесте за "
        f"{N_INFERENCE_REPEATS} повторов после прогрева. Абсолютные миллисекунды",
        "зависят от CPU; между моделями сравнивать нужно один и тот же запуск.",
        "",
        "СЖАТИЕ: НАКОПЛЕННАЯ ЭНЕРГИЯ",
        "---------------------------",
        *spectrum_rows,
        "",
        "МЕТРИКИ ПО КОМПОНЕНТАМ",
        "----------------------",
        *per_label_csv,
        "",
        "БАЛАНС МЕТОК В ДАТАСЕТЕ",
        "-----------------------",
        *balance_lines,
        "",
        "ПРИМЕРЫ ПРОМПТОВ",
        "----------------",
        *example_lines,
        "",
        "ХАРАКТЕРНЫЕ ТОКЕНЫ МЕТОК (средний TF-IDF, train, до SVD)",
        "--------------------------------------------------------",
        *token_lines,
        "",
        "ПЕРВЫЕ ЛАТЕНТНЫЕ КОМПОНЕНТЫ (модуль нагрузки; знак условен)",
        "-----------------------------------------------------------",
        *svd_lines,
        "",
        "ПРОТОКОЛ",
        "--------",
        "Целевые переменные — четыре бинарные колонки. Модель каждого класса",
        "обучается по схеме one-vs-rest. Признаки классификаторов — первые",
        f"{SVD_COMPONENTS} компонент TruncatedSVD, стандартизованные по train.",
        "TF-IDF: unigram-леммы, max_features = 1000, min_df = 2, нормировка l2.",
        "LogisticRegression: L2, C = 1, solver = lbfgs.",
        "SVC: ядро RBF, C = 1, gamma = scale, probability = True.",
        "TruncatedSVD не центрирует разреженную матрицу, поэтому F_k — это",
        "explained_variance_ratio_ sklearn (латентный семантический анализ),",
        "а не доля дисперсии центрированного PCA.",
        "",
        "ВЕРСИИ БИБЛИОТЕК",
        "----------------",
        versions,
        "",
    ])
    return "\n".join(parts) + "\n"


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    _configure_stdout()
    _apply_poster_style()
    # В scikit-learn 1.9 probability=True помечен устаревшим в пользу CalibratedClassifierCV.
    # Параметр оставлен: в протоколе эксперимента у SVC нужна калибровка Платта.
    warnings.filterwarnings(
        "ignore",
        message="The `probability` parameter was deprecated",
        category=FutureWarning,
    )
    _check_paradigms()
    _audit_signal_leak()

    rng = np.random.default_rng(RANDOM_STATE)

    print("[1/5] Генерация синтетических промптов CapsuleOS")
    dataset = generate_dataset(N_PROMPTS, rng)
    dataset.to_csv(PATH_DATASET, index=False, encoding="utf-8-sig")
    balance = dataset[LABELS].sum()
    print(f"      объектов: {len(dataset)}; файл: {PATH_DATASET.name}")
    for label in LABELS:
        print(f"      {label:<16} {int(balance[label]):>4} положительных")

    print("[2/5] Предобработка текста и матрица TF-IDF")
    prompts = dataset["prompt"].tolist()
    targets = dataset[LABELS].to_numpy(dtype=int)
    # Стратификация по комбинации меток сохраняет состав UI-экранов в обеих частях.
    stratify_key = dataset[LABELS].astype(str).agg("".join, axis=1).to_numpy()
    index = np.arange(len(dataset))
    train_index, test_index = train_test_split(
        index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_key,
    )
    prompts_train = [prompts[i] for i in train_index]
    prompts_test = [prompts[i] for i in test_index]
    y_train = targets[train_index]
    y_test = targets[test_index]

    vectorizer = TfidfVectorizer(
        analyzer=preprocess,
        max_features=TFIDF_MAX_FEATURES,
        min_df=2,
    )
    x_train_tfidf = vectorizer.fit_transform(prompts_train)
    x_test_tfidf = vectorizer.transform(prompts_test)
    feature_names = np.array(vectorizer.get_feature_names_out())
    n_features = int(x_train_tfidf.shape[1])
    print(f"      словарь ограничен p = {TFIDF_MAX_FEATURES}, фактически p = {n_features}")
    print(f"      train/test = {len(train_index)}/{len(test_index)}, min_df = 2")
    if n_features <= SVD_SPECTRUM:
        raise RuntimeError(
            f"Для спектра из {SVD_SPECTRUM} компонент нужно p > {SVD_SPECTRUM}, сейчас p = {n_features}."
        )

    print("[3/5] TruncatedSVD и накопленная объяснённая дисперсия")
    svd = TruncatedSVD(n_components=SVD_SPECTRUM, random_state=RANDOM_STATE)
    latent_train_full = svd.fit_transform(x_train_tfidf)
    latent_test_full = svd.transform(x_test_tfidf)
    cumulative = np.cumsum(svd.explained_variance_ratio_)
    energy_at_k = float(cumulative[SVD_COMPONENTS - 1])
    # Компоненты упорядочены по сингулярным значениям, поэтому срез [:k] — это и есть решение при k = 30.
    latent_train = latent_train_full[:, :SVD_COMPONENTS]
    latent_test = latent_test_full[:, :SVD_COMPONENTS]
    print(f"      F_30 = {energy_at_k:.4f} ({energy_at_k * 100:.1f}% энергии при k = {SVD_COMPONENTS})")
    for k in (10, 20, 30, 40, 50):
        print(f"      F_{k:<2} = {cumulative[k - 1]:.4f}")

    scaler = StandardScaler()
    z_train = scaler.fit_transform(latent_train)
    z_test = scaler.transform(latent_test)

    print("[4/5] Обучение классификаторов в латентном пространстве")
    logreg = OneVsRestClassifier(
        LogisticRegression(solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE),
    )
    svc = OneVsRestClassifier(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    )
    models = {
        "LogisticRegression": logreg,
        "SVC (RBF)": svc,
    }
    evaluated = []
    per_label_rows = []
    for name, model in models.items():
        print(f"      обучение: {name}")
        model.fit(z_train, y_train)
        pred = model.predict(z_test)
        scores = _positive_scores(model, z_test)
        if scores.shape != y_test.shape:
            raise RuntimeError(f"{name}: ожидалась матрица вероятностей {y_test.shape}, получено {scores.shape}")
        accuracy = _label_accuracy(y_test, pred)
        subset = float(accuracy_score(y_test, pred))
        roc_auc = float(roc_auc_score(y_test, scores, average="macro"))
        per_auc = roc_auc_score(y_test, scores, average=None)
        per_acc = _per_label_accuracy(y_test, pred)
        infer_ms = _measure_predict_ms(model, z_test, N_INFERENCE_REPEATS)
        evaluated.append({
            "Модель": name,
            "Accuracy": accuracy,
            "SubsetAccuracy": subset,
            "ROC-AUC": roc_auc,
            "Время_мс": infer_ms,
            "мс_на_объект": infer_ms / len(z_test),
        })
        for index, label in enumerate(LABELS):
            per_label_rows.append({
                "Модель": name,
                "Компонент": label,
                "Accuracy": float(per_acc[index]),
                "ROC-AUC": float(per_auc[index]),
            })
        print(
            f"      {name}: Accuracy = {accuracy:.3f}, "
            f"Subset = {subset:.3f}, ROC-AUC = {roc_auc:.3f}, "
            f"инференс = {infer_ms:.2f} мс"
        )

    summary = pd.DataFrame(evaluated)
    per_label = pd.DataFrame(per_label_rows)

    print("[5/5] Графики и отчёт для постера")
    plot_explained_variance(cumulative, energy_at_k, PATH_VARIANCE)
    plot_model_comparison(summary, PATH_COMPARISON)

    token_map = _top_tfidf_tokens(x_train_tfidf, y_train, feature_names)
    token_lines = [
        f"  {LABELS_RU[label]}: {', '.join(tokens)}" for label, tokens in token_map.items()
    ]
    svd_lines = _top_svd_terms(svd, feature_names)

    showcase_mask = dataset["prompt"].isin([sample["prompt"] for sample in CANONICAL])
    examples = pd.concat([
        dataset.loc[showcase_mask],
        dataset.loc[~showcase_mask].sample(4, random_state=RANDOM_STATE),
    ], ignore_index=True)

    versions = "\n".join([
        f"python {sys.version.split()[0]}",
        f"pandas {pd.__version__}",
        f"numpy {np.__version__}",
        f"matplotlib {matplotlib.__version__}",
        f"scikit-learn {__import__('sklearn').__version__}",
    ])
    report = build_report(
        n_train=len(train_index),
        n_test=len(test_index),
        n_features=n_features,
        cumulative=cumulative,
        summary=summary,
        per_label=per_label,
        balance=balance,
        examples=examples,
        token_lines=token_lines,
        svd_lines=svd_lines,
        versions=versions,
    )
    PATH_REPORT.write_text(report, encoding="utf-8-sig")

    print()
    print(summary.to_string(index=False, formatters={
        "Accuracy": "{:.3f}".format,
        "SubsetAccuracy": "{:.3f}".format,
        "ROC-AUC": "{:.3f}".format,
        "Время_мс": "{:.2f}".format,
        "мс_на_объект": "{:.4f}".format,
    }))
    print()
    print(
        f"Пространство: TF-IDF p = {n_features} -> SVD k = {SVD_COMPONENTS}, "
        f"F_30 = {energy_at_k:.3f}"
    )
    print(f"Спектр:     {PATH_VARIANCE}")
    print(f"Сравнение:  {PATH_COMPARISON}")
    print(f"Цифры:      {PATH_REPORT}")
    print()
    print(report)


if __name__ == "__main__":
    main()
