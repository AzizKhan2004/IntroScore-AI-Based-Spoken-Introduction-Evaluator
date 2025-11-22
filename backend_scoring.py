# backend/scoring.py

from typing import Dict, Any, List, Tuple
import re
import numpy as np

from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from backend_rubric_config import (
    CRITERIA,
    CriterionConfig,
    FILLER_RATE_BANDS,
    SENTIMENT_BANDS,
    TOTAL_WEIGHT,
)


_sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_sentiment_analyzer = SentimentIntensityAnalyzer()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize_words(text: str) -> List[str]:
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return [w for w in text.split() if w]


def split_sentences(text: str) -> List[str]:
    # basic split on '.', '?', '!'
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def compute_basic_stats(text: str) -> Dict[str, int]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
    }


def keyword_presence_score(text: str, keywords: List[str]) -> Tuple[float, List[str]]:
    """
    Returns (score_0_to_1, found_keywords)
    Score = fraction of keywords (or key-phrases) that appear at least once.
    """
    text_norm = text.lower()
    found = []
    for kw in keywords:
        if kw.lower() in text_norm:
            found.append(kw)
    if not keywords:
        return 0.0, []
    coverage = len(found) / len(keywords)
    return coverage, found


def semantic_similarity_score(text: str, description: str) -> float:
    """
    Returns cosine similarity between transcript and rubric description, mapped to [0,1].
    """
    embeddings = _sentence_model.encode([text, description], convert_to_tensor=True)
    sim = util.cos_sim(embeddings[0], embeddings[1]).item()  # [-1, 1]
    # Map from [-1, 1] → [0, 1]
    return (sim + 1) / 2


def flow_score(text: str) -> Dict[str, Any]:
    """
    Very simple heuristic:
    - Check relative order of:
      greeting → self → details → closing
    We use keyword index positions.
    """
    text_norm = text.lower()

    def first_index_of_any(subs: List[str]) -> int:
        idxs = []
        for s in subs:
            i = text_norm.find(s)
            if i != -1:
                idxs.append(i)
        return min(idxs) if idxs else -1

    greeting_idx = first_index_of_any(["hello", "hi", "good morning", "good afternoon", "good evening"])
    self_idx = first_index_of_any(["my name is", "myself", "i am"])
    details_idx = first_index_of_any(["family", "hobby", "hobbies", "interest", "subject", "goal", "dream"])
    closing_idx = first_index_of_any(["thank you", "thanks"])

    # Score based on correct ordering
    order_score = 0
    steps = []

    if greeting_idx != -1:
        steps.append("greeting")
    if self_idx != -1:
        steps.append("self")
    if details_idx != -1:
        steps.append("details")
    if closing_idx != -1:
        steps.append("closing")

    # ideal order: greeting < self < details < closing
    # we just check monotonic increasing indices when they exist
    indices = [x for x in [greeting_idx, self_idx, details_idx, closing_idx] if x != -1]

    if len(indices) >= 2:
        # count number of monotonic increases
        inc = sum(1 for i in range(len(indices) - 1) if indices[i] < indices[i + 1])
        order_score = inc / (len(indices) - 1)
    else:
        order_score = 0.3 if steps else 0.0  # very weak signal if we barely find anything

    return {
        "score": order_score,
        "steps_detected": steps
    }


def length_score(word_count: int, min_words: int, max_words: int) -> float:
    """
    If within range -> 1.
    If outside, decrease score linearly.
    """
    if min_words is None or max_words is None:
        return 1.0

    if word_count < min_words:
        # 0 at 0 words, 1 at min_words
        return max(0.0, word_count / min_words)
    if word_count > max_words:
        # 0 at 2x max_words, 1 at max_words
        if word_count >= 2 * max_words:
            return 0.0
        over = word_count - max_words
        span = max_words
        return max(0.0, 1 - (over / span))
    return 1.0


def filler_rate_score(text: str, filler_words: List[str]) -> Dict[str, Any]:
    words = tokenize_words(text)
    if not words:
        return {"score": 0.0, "rate_percent": 0.0, "filler_count": 0, "total_words": 0}

    word_str = " " + " ".join(words) + " "
    filler_count = 0
    filler_found = []
    for f in filler_words:
        pattern = r"\b" + re.escape(f.lower()) + r"\b"
        matches = re.findall(pattern, word_str.lower())
        if matches:
            filler_count += len(matches)
            filler_found.append(f)

    rate = (filler_count / len(words)) * 100

    # Map to band
    band_factor = 0.2  # worst default
    for threshold, factor in FILLER_RATE_BANDS:
        if rate <= threshold:
            band_factor = factor
            break

    return {
        "score": band_factor,
        "rate_percent": rate,
        "filler_count": filler_count,
        "total_words": len(words),
        "filler_found": filler_found
    }


def sentiment_score(text: str) -> Dict[str, Any]:
    vs = _sentiment_analyzer.polarity_scores(text)
    compound = vs["compound"]  # -1 to 1

    # Map to [0,1] factor via bands
    factor = 0.2
    for threshold, band_factor in SENTIMENT_BANDS:
        if compound >= threshold:
            factor = band_factor
            break

    return {
        "score": factor,
        "compound": compound,
        "raw": vs
    }


def score_transcript(transcript: str) -> Dict[str, Any]:
    transcript = transcript.strip()
    basic_stats = compute_basic_stats(transcript)
    word_count = basic_stats["word_count"]
    sentence_count = basic_stats["sentence_count"]

    criteria_results = []
    weighted_sum = 0.0

    for crit in CRITERIA:
        detail: Dict[str, Any] = {
            "id": crit.id,
            "category": crit.category,
            "name": crit.name,
            "weight": crit.weight,
        }

        if crit.type == "keyword_presence":
            rule_score, found = keyword_presence_score(transcript, crit.keywords or [])
            semantic = semantic_similarity_score(transcript, crit.description)
            combined = 0.5 * rule_score + 0.5 * semantic
            detail.update({
                "rule_score": round(rule_score, 3),
                "semantic_similarity": round(semantic, 3),
                "combined_score": round(combined, 3),
                "keywords_found": found,
            })
            crit_score = combined

        elif crit.type == "semantic":
            semantic = semantic_similarity_score(transcript, crit.description)
            detail.update({
                "semantic_similarity": round(semantic, 3),
            })
            crit_score = semantic

        elif crit.type == "flow":
            flow = flow_score(transcript)
            crit_score = flow["score"]
            detail.update(flow)

        elif crit.type == "length":
            ls = length_score(word_count, crit.min_words, crit.max_words)
            crit_score = ls
            detail.update({
                "length_score": round(ls, 3),
                "min_words": crit.min_words,
                "max_words": crit.max_words,
            })

        elif crit.type == "filler_rate":
            fr = filler_rate_score(transcript, crit.extra.get("filler_words", []))
            crit_score = fr["score"]
            detail.update(fr)

        elif crit.type == "sentiment":
            sent = sentiment_score(transcript)
            crit_score = sent["score"]
            detail.update(sent)

        else:
            crit_score = 0.0
            detail["note"] = "Unsupported criterion type."

        weighted = crit_score * crit.weight
        weighted_sum += weighted

        detail["score_0_to_1"] = round(crit_score, 3)
        detail["score_weighted"] = round(weighted, 3)
        criteria_results.append(detail)

    overall_score_0_to_1 = weighted_sum / TOTAL_WEIGHT if TOTAL_WEIGHT > 0 else 0.0
    overall_score_0_to_100 = round(overall_score_0_to_1 * 100, 1)

    return {
        "transcript": transcript,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "overall_score": overall_score_0_to_100,
        "overall_score_0_to_1": round(overall_score_0_to_1, 3),
        "criteria": criteria_results,
    }
