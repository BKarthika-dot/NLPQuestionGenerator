"""Explainable, non-neural evaluation for short and long-form answers."""

import math
import re
from collections import Counter
from typing import Any, Iterable, List, Optional


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "which", "with", "what", "when", "where", "who", "why",
    "how", "into", "than", "these", "those", "they", "them", "can", "may",
}

SYNONYMS = {
    "sunlight": "light",
    "sunshine": "light",
    "produce": "convert",
    "create": "convert",
    "generate": "convert",
    "stored": "store",
    "storing": "store",
}


def _normalize_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "sis")):
        return token[:-1]
    return SYNONYMS.get(token, token)


def tokenize(text: str, *, remove_stop_words: bool = True) -> List[str]:
    tokens = [_normalize_token(token) for token in re.findall(r"[a-z0-9]+", (text or "").lower())]
    return [token for token in tokens if not remove_stop_words or token not in STOP_WORDS]


def _term_frequency(tokens: Iterable[str]) -> Counter:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return Counter({term: count / total for term, count in counts.items()})


def tfidf_cosine(reference_tokens: List[str], student_tokens: List[str]) -> float:
    """Compute TF-IDF cosine for the two documents without a model."""
    if not reference_tokens or not student_tokens:
        return 0.0
    documents = [set(reference_tokens), set(student_tokens)]
    vocabulary = documents[0] | documents[1]
    reference_tf = _term_frequency(reference_tokens)
    student_tf = _term_frequency(student_tokens)
    reference_vector = {}
    student_vector = {}
    for term in vocabulary:
        document_frequency = sum(term in document for document in documents)
        inverse_document_frequency = math.log(3 / (1 + document_frequency)) + 1
        reference_vector[term] = reference_tf[term] * inverse_document_frequency
        student_vector[term] = student_tf[term] * inverse_document_frequency
    dot = sum(reference_vector[term] * student_vector[term] for term in vocabulary)
    reference_norm = math.sqrt(sum(value * value for value in reference_vector.values()))
    student_norm = math.sqrt(sum(value * value for value in student_vector.values()))
    return dot / (reference_norm * student_norm) if reference_norm and student_norm else 0.0


def _jaccard(left: set, right: set) -> float:
    return len(left & right) / len(left | right) if left | right else 0.0


def _phrase_coverage(phrases: Iterable[str], student_tokens: set) -> float:
    phrases = list(phrases)
    if not phrases:
        return 0.0
    covered = 0
    for phrase in phrases:
        phrase_tokens = set(tokenize(phrase))
        if phrase_tokens and len(phrase_tokens & student_tokens) / len(phrase_tokens) >= 0.5:
            covered += 1
    return covered / len(phrases)


def _ngrams(tokens: List[str], size: int = 2) -> set:
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def _derive_keywords(reference_tokens: List[str], limit: int = 12) -> List[str]:
    return [term for term, _ in Counter(reference_tokens).most_common(limit)]


def _grade(score: float) -> str:
    if score >= 0.80:
        return "Excellent"
    if score >= 0.65:
        return "Good"
    if score >= 0.45:
        return "Partial"
    return "Incorrect"


def evaluate_answer(
    student_answer: str,
    reference_answer: str,
    keywords: Optional[Iterable[str]] = None,
    concepts: Optional[Iterable[str]] = None,
) -> dict:
    """Return explainable component scores and feedback in the range 0..1."""
    reference_tokens = tokenize(reference_answer)
    student_tokens = tokenize(student_answer)
    reference_set = set(reference_tokens)
    student_set = set(student_tokens)
    keyword_list = list(keywords or _derive_keywords(reference_tokens))
    keyword_tokens = set(token for keyword in keyword_list for token in tokenize(keyword))
    keyword_score = len(keyword_tokens & student_set) / len(keyword_tokens) if keyword_tokens else 0.0
    tfidf_score = tfidf_cosine(reference_tokens, student_tokens)
    word_overlap = _jaccard(reference_set, student_set)
    ngram_score = _jaccard(_ngrams(reference_tokens), _ngrams(student_tokens))
    semantic_score = min(1.0, 0.75 * tfidf_score + 0.15 * word_overlap + 0.10 * ngram_score)
    concept_score = _phrase_coverage(concepts or [], student_set) if concepts else keyword_score
    expected_length = max(len(reference_tokens), 1)
    length_score = min(1.0, len(student_tokens) / expected_length) if student_tokens else 0.0
    relevance_score = min(1.0, 0.7 * semantic_score + 0.3 * length_score)
    final_score = min(1.0, max(0.0, 0.40 * semantic_score + 0.30 * keyword_score + 0.20 * concept_score + 0.10 * relevance_score))
    grade = _grade(final_score)

    feedback = []
    if keyword_score >= 0.75:
        feedback.append("Good coverage of the main terms.")
    else:
        missing = [keyword for keyword in keyword_list if not set(tokenize(keyword)) & student_set]
        if missing:
            feedback.append(f"Consider mentioning: {', '.join(missing[:3])}.")
    if semantic_score < 0.45:
        feedback.append("The answer is not yet closely aligned with the reference answer.")
    elif concept_score >= 0.75:
        feedback.append("The key concepts are covered.")
    if length_score < 0.45:
        feedback.append("Add more explanation to fully answer the question.")
    if not feedback:
        feedback.append("The answer covers some of the reference content; add detail for a stronger response.")

    return {
        "similarity": round(semantic_score, 4),
        "similarity_score": round(semantic_score, 4),
        "keyword_score": round(keyword_score, 4),
        "concept_score": round(concept_score, 4),
        "relevance_score": round(relevance_score, 4),
        "score": round(final_score * 10),
        "overall_score": round(final_score * 100),
        "grade": grade,
        "correct": final_score >= 0.75,
        "threshold": 0.75,
        "feedback": feedback,
    }