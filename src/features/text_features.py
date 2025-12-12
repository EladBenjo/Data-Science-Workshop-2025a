"""Text processing utilities extracted from the FOMC speeches notebook."""
from __future__ import annotations

from typing import Iterable, List, Sequence

from nltk.tokenize import word_tokenize
import pandas as pd

DEFAULT_CONTRASTIVE_KEYWORDS: tuple[str, ...] = (
    "however",
    "but",
    "although",
    "yet",
    "nevertheless",
    "nonetheless",
    "still",
)


DEFAULT_ECONOMIC_KEYWORDS = {
    "Inflation & Expectations": [
        "inflation expectation",
        "core CPI",
        "core PCE",
        "inflationary pressure",
        "price stability",
        "headline inflation",
        "cost-push inflation",
        "demand-pull inflation",
        "supply bottlenecks",
        "input costs",
        "commodity prices",
    ],
    "Interest Rates & Monetary Policy": [
        "interest rate",
        "bank rate",
        "fund rate",
        "monetary policy",
        "quantitative tightening",
        "balance sheet reduction",
        "liquidity",
        "monetary tightening",
        "financial conditions",
        "forward guidance",
        "neutral rate",
        "real interest rate",
        "hawkish",
        "dovish",
        "policy stance",
        "accommodative",
        "restrictive",
    ],
    "Labor Market & Economic Growth": [
        "employment",
        "unemployment",
        "job market",
        "growth",
        "productivity",
        "labor force participation",
        "wage inflation",
        "wage growth",
        "jobless claims",
        "payroll",
        "hiring freeze",
        "soft landing",
        "consumer confidence",
        "household spending",
        "business investment",
    ],
    "Markets & Financial Stability": [
        "exchange rate",
        "deficit",
        "demand",
        "credit spreads",
        "treasury yield",
        "bond market expectations",
        "financial stability",
        "market volatility",
        "yield curve inversion",
        "capital expenditures",
        "corporate borrowing",
        "debt servicing costs",
    ],
}


def calculate_ttr(text: str) -> float:
    """Calculate the type-token ratio (TTR) for lexical richness."""
    tokens = word_tokenize(text)
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) if tokens else 0.0


def split_sentences_custom(
    text: str,
    nlp,
    contrastive_keywords: Sequence[str] | None = None,
) -> List[str]:
    """Split text into sentences and further divide on contrastive keywords."""
    contrastive_keywords = tuple(contrastive_keywords or DEFAULT_CONTRASTIVE_KEYWORDS)
    doc = nlp(text)
    processed: list[str] = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        found_keyword = None
        for word in contrastive_keywords:
            if f" {word} " in sentence_text.lower():
                found_keyword = word
                break
        if found_keyword:
            parts = sentence_text.split(found_keyword, 1)
            first_part = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                second_part = f"{found_keyword} {parts[1].strip()}"
                processed.extend([first_part, second_part])
            else:
                processed.append(first_part)
        else:
            processed.append(sentence_text)
    return processed


def flatten_keywords(keyword_dict: dict[str, Iterable[str]]) -> set[str]:
    return set(word.lower() for values in keyword_dict.values() for word in values)


def filter_target_sentences(sentences: Iterable[str], keywords: Iterable[str]) -> list[str]:
    keyword_set = {kw.lower() for kw in keywords}
    return [sent for sent in sentences if any(word in sent.lower() for word in keyword_set)]


def calculate_ttr_for_sentences(sentences: Iterable[str]) -> float | None:
    full_text = " ".join(sentences)
    return calculate_ttr(full_text) if full_text else None


def add_text_features(
    df: pd.DataFrame,
    text_column: str,
    nlp,
    keyword_dict: dict[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    """Augment a DataFrame with text-derived features."""
    keyword_dict = keyword_dict or DEFAULT_ECONOMIC_KEYWORDS
    keywords = flatten_keywords(keyword_dict)

    def _split(text: str) -> list[str]:
        return split_sentences_custom(text, nlp)

    df = df.copy()
    df["split_sentences"] = df[text_column].apply(_split)
    df["sentence_count"] = df["split_sentences"].apply(len)
    df["target_sentences"] = df["split_sentences"].apply(lambda sents: filter_target_sentences(sents, keywords))
    df["target_sentence_count"] = df["target_sentences"].apply(len)
    df["target_sentence_ratio"] = df.apply(
        lambda row: row["target_sentence_count"] / row["sentence_count"] if row["sentence_count"] else 0.0,
        axis=1,
    )
    df["ttr_full"] = df[text_column].apply(calculate_ttr)
    df["ttr_filtered"] = df["target_sentences"].apply(calculate_ttr_for_sentences)
    df["speech_length"] = df[text_column].str.len()
    return df


__all__ = [
    "DEFAULT_CONTRASTIVE_KEYWORDS",
    "DEFAULT_ECONOMIC_KEYWORDS",
    "calculate_ttr",
    "split_sentences_custom",
    "flatten_keywords",
    "filter_target_sentences",
    "calculate_ttr_for_sentences",
    "add_text_features",
]
