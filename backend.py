# ============================================================
# backend.py
#
# PDF -> page-aware extraction -> cleaning -> semantic chunking
# -> embeddings -> topic retrieval -> question generation
# -> [RAG] question-specific retrieval -> long-form answer generation
# -> final question + system-wide RAG evaluation score
# ============================================================

import hashlib
import re
import logging
from pathlib import Path
from threading import Lock
from typing import List, Dict, Optional, Any

import numpy as np
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

logger = logging.getLogger("question-lab.backend")


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Semantic chunking
SEMANTIC_SIMILARITY_THRESHOLD = 0.45
MIN_SENTENCES_PER_CHUNK = 2
MAX_SENTENCES_PER_CHUNK = 6

# Retrieval
TOPIC_TOP_K = 6

# Chunks retrieved specifically to answer a single question (this is the
# dedicated RAG step: question -> retrieval -> generation). Larger than the
# old QUESTION_TOP_K=4 so a long-form answer actually has enough grounding
# material to draw a complete, multi-sentence answer from.
ANSWER_TOP_K = 6

# Generation
# NOTE: flan-t5 is fine-tuned mostly on <=512 token inputs.
MAX_INPUT_TOKENS = 512
MAX_QUESTION_TOKENS = 64

# Long-form answers need real room to breathe. 48 tokens was hard-capping
# every answer to a single short clause regardless of how much the question
# actually required to answer correctly.
MAX_ANSWER_TOKENS = 220
MIN_ANSWER_TOKENS = 24

# How many times to retry answer generation for the SAME question (with the
# SAME retrieved context) before giving up on that question and generating
# a fresh one instead.
ANSWER_RETRIES_PER_QUESTION = 2

# How many chunks either side of a seed chunk to pull in as local context
# when generating a question, so the model has enough surrounding text
# without dragging in the whole document.
QUESTION_CONTEXT_WINDOW = 1

# Two questions are treated as near-duplicates (and the second rejected) if
# their embedding cosine similarity is at or above this. Catches paraphrases
# like "What is the basic layer...?" vs "...called?" vs "...structure?"
# that don't match as exact strings but are the same question.
QUESTION_DEDUP_THRESHOLD = 0.85

# A question is rejected before any answer is even attempted if its best
# retrieval match isn't actually close to it semantically -- this is the
# cheap "is this question well-grounded in the document at all" check.
# Incoherent/malformed questions (mismatched clauses fused from unrelated
# sentences, e.g. "Which religions had more than one main religion?")
# typically fail this because their phrasing doesn't correspond to any
# real passage, even though they're grammatically a valid question.
MIN_QUESTION_CONTEXT_RELEVANCE = 0.30

QUESTION_TYPE = "Long Answer"
DIFFICULTIES = {"easy", "medium", "hard"}

# ------------------------------------------------------------
# Grounded answer grading (NLI cross-encoder + sentence embeddings)
#
# Nothing in this grader looks at surface wording: no keyword overlap, no
# n-gram/ROUGE/BLEU, no TF-IDF. Every decision is made either by an NLI
# cross-encoder (does text A entail / contradict text B?) or by cosine
# similarity between sentence embeddings, so a correct paraphrase scores the
# same as a verbatim copy, and a fluent answer that says the opposite does not.
# ------------------------------------------------------------

# Tried in order. The first is the stronger model; its tokenizer needs
# `pip install sentencepiece`. If it can't load for any reason, the second
# (plain WordPiece tokenizer, no extra dependency) is used instead.
NLI_MODEL_CANDIDATES = [
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-MiniLM2-L6-H768",
]
NLI_MAX_TOKENS = 320
NLI_BATCH_SIZE = 16

# Final grade (0-1) at/above which an answer counts as correct.
GRADE_CORRECT_THRESHOLD = 0.60

# How many source sentences are offered to the NLI model as evidence for one claim.
EVIDENCE_TOP_K = 3
MAX_CONTEXT_SENTENCES = 40
MAX_KEY_POINTS = 6
MAX_STUDENT_CLAIMS = 8
MAX_STUDENT_CHARS = 2500
PROFILE_CACHE_SIZE = 512

# A reference-answer sentence only becomes a graded "key point" if the source
# text entails it at least this strongly -- the generated reference can be
# wrong, the source text is the ground truth.
KEYPOINT_GROUNDING_MIN = 0.40
KEYPOINT_DEDUP_SIM = 0.92

# Student sentences less related than this to both the question and the
# source are treated as filler: they neither earn nor lose marks.
CLAIM_RELEVANCE_MIN = 0.30
# Skip running NLI on pairs that share almost no meaning (saves compute).
NLI_MIN_RELEVANCE = 0.25
# A contradiction only counts between texts that are actually about the same thing.
CONTRADICTION_MIN_RELEVANCE = 0.35
CONTRADICTION_THRESHOLD = 0.60
CONTRADICTION_PENALTY = 0.75
# If some source sentence clearly entails a statement, the statement is supported.
# Contradiction signals from OTHER sentences (typically a near-identical sentence
# about a different subject, e.g. "unsupervised" vs "supervised") are then ignored.
ENTAILMENT_OVERRIDE = 0.50

# Partial credit: a key point the student paraphrased closely but did not fully
# entail earns up to this fraction of a mark, scaled by embedding similarity.
# It can never override a contradiction.
SOFT_CREDIT_WEIGHT = 0.50
SOFT_SIM_FLOOR = 0.50
SOFT_SIM_CEIL = 0.90

# How much the "is every claim backed by the source" check can lower a grade.
SUPPORT_WEIGHT = 0.30
# An answer this close to the question (and not longer) is just restating it.
ECHO_SIMILARITY = 0.92

print(f"[INFO] Device: {DEVICE}")


# ============================================================
# 1. PDF LOADING
# ============================================================

def load_pdf(file_path: str) -> List[Document]:
    logger.info("load_pdf: start path=%s", file_path)
    if not Path(file_path).exists():
        raise ValueError(f"PDF does not exist: {file_path}")

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        raise ValueError(f"Could not read PDF: {str(e)}")

    result = []
    for index, document in enumerate(documents, start=1):
        text = document.page_content or ""
        if not text.strip():
            continue
        metadata = dict(document.metadata)
        metadata["page"] = index
        metadata["source_file"] = Path(file_path).name
        result.append(Document(page_content=text, metadata=metadata))

    if not result:
        raise ValueError("No readable text was found in the PDF.")

    print(f"[INFO] Read {len(result)} readable pages.")
    logger.info("load_pdf: complete readable_pages=%d", len(result))
    return result


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bPage\s+\d+\s*(?:of\s+\d+)?\b", " ", text, flags=re.IGNORECASE)
    return text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    logger.info("clean_documents: start documents=%d", len(documents))
    result = []
    for document in documents:
        text = clean_text(document.page_content)
        if not text:
            continue
        result.append(Document(page_content=text, metadata=document.metadata))
    logger.info("clean_documents: complete documents=%d", len(result))
    return result


# ============================================================
# 3. SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []

    protected = re.sub(
        r"\b(e\.g|i\.e|Mr|Mrs|Dr|etc)\.",
        lambda m: m.group(0).replace(".", "<DOT>"),
        text,
        flags=re.IGNORECASE,
    )

    sentences = re.split(r"(?<=[.!?])\s+", protected)

    result = []
    for sentence in sentences:
        sentence = sentence.replace("<DOT>", ".").strip()
        if sentence:
            result.append(sentence)
    return result


# ============================================================
# 4. SECTION DETECTION
# ============================================================

# ============================================================
# 4b. BOILERPLATE / ADMINISTRATIVE TEXT DETECTION
# ============================================================

# Matches the kind of front-matter and administrative text that makes a
# technically-valid but useless question ("What is the course code for
# NLP?"). These chunks still exist in the index and can be retrieved as
# supporting context, but they're excluded from the pool of chunks a
# question is allowed to be generated FROM.
BOILERPLATE_PATTERNS = re.compile(
    r"\b(course\s*code|instructor|credits?|isbn|table of contents|"
    r"syllabus|prerequisites?|copyright|all rights reserved|"
    r"student\s*id|roll\s*no|semester|academic\s*year|revision\s*history|"
    r"version\s*\d|published\s*by)\b",
    flags=re.IGNORECASE,
)

MIN_SEEDABLE_WORDS = 12


def is_boilerplate_chunk(text: str) -> bool:
    """
    True if a chunk looks like administrative/front-matter text rather
    than substantive content -- either it matches a known admin-text
    pattern, or it's too short/sparse to carry a real concept (a course
    code line, a header, a lone date).
    """
    if not text:
        return True
    if BOILERPLATE_PATTERNS.search(text):
        return True
    if len(text.split()) < MIN_SEEDABLE_WORDS:
        return True
    return False


def detect_section(text: str) -> str:
    if not text:
        return "General"

    match = re.match(
        r"^\s*(\d+(?:\.\d+)*[\.\)]?\s+[A-Z][A-Za-z0-9 ,:&\-_]{2,100})", text
    )
    if match:
        heading = match.group(1).strip()
        if len(heading) <= 120:
            return heading

    match = re.match(
        r"^\s*((?:Chapter|Section)\s+\d+(?:\s*[:\-]\s*[A-Za-z][A-Za-z0-9 ,:&\-_]{0,100})?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    first_part = text[:100].strip()
    if first_part.isupper() and len(first_part) <= 100:
        return first_part

    return "General"


# ============================================================
# 5. SEMANTIC EMBEDDING MODEL
# ============================================================

class SemanticEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.model.max_seq_length = 256
        print("[INFO] Embedding model loaded.")

    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 384))
        return self.model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
        )

    def similarity(self, query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.matmul(document_embeddings, query_embedding)


# ============================================================
# 6. SEMANTIC CHUNKER
# ============================================================

class SemanticChunker:
    def __init__(self, embedder: SemanticEmbedder, threshold: float = SEMANTIC_SIMILARITY_THRESHOLD):
        self.embedder = embedder
        self.threshold = threshold

    def create_chunk(self, sentences: List[str], metadata: Dict[str, Any], chunk_index: int) -> Document:
        text = " ".join(sentences).strip()
        chunk_metadata = {
            **metadata,
            "section": detect_section(text),
            "chunk_id": None,
            "chunk_index_on_page": chunk_index,
            "semantic_chunk": True,
            "sentence_count": len(sentences),
            "chunk_length": len(text),
        }
        return Document(page_content=text, metadata=chunk_metadata)

    def chunk_page(self, document: Document) -> List[Document]:
        sentences = split_into_sentences(document.page_content)
        if not sentences:
            return []

        if len(sentences) <= 2:
            return [self.create_chunk(sentences, document.metadata, 0)]

        embeddings = self.embedder.encode(sentences)
        chunks = []
        current = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = torch.dot(embeddings[i - 1], embeddings[i]).item()
            semantic_boundary = similarity < self.threshold
            size_boundary = len(current) >= MAX_SENTENCES_PER_CHUNK

            if (semantic_boundary and len(current) >= MIN_SENTENCES_PER_CHUNK) or size_boundary:
                chunks.append(self.create_chunk(current, document.metadata, len(chunks)))
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            chunks.append(self.create_chunk(current, document.metadata, len(chunks)))

        return chunks

    def split(self, documents: List[Document]) -> List[Document]:
        logger.info("SemanticChunker.split: start documents=%d", len(documents))
        all_chunks = []
        chunk_id = 0
        for document in documents:
            for chunk in self.chunk_page(document):
                chunk.metadata["chunk_id"] = chunk_id
                all_chunks.append(chunk)
                chunk_id += 1

        print(f"[INFO] Semantic chunks: {len(all_chunks)}")
        logger.info("SemanticChunker.split: complete chunks=%d", len(all_chunks))
        return all_chunks


# ============================================================
# 7. SEMANTIC INDEX
# ============================================================

class SemanticIndex:
    def __init__(self, chunks: List[Document], embedder: SemanticEmbedder):
        self.chunks = chunks
        self.embedder = embedder
        print("[INFO] Creating embedding index...")
        self.embeddings = embedder.encode([chunk.page_content for chunk in chunks])
        print("[INFO] Embedding index ready.")

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        query_embedding = self.embedder.encode([query])[0]
        scores = self.embedder.similarity(query_embedding, self.embeddings)
        top_k = min(top_k, len(self.chunks))
        values, indices = torch.topk(scores, k=top_k)

        return [
            {"chunk": self.chunks[index], "score": float(score)}
            for score, index in zip(values.tolist(), indices.tolist())
        ]


# ============================================================
# 8. CONTEXT RETRIEVER
# ============================================================

class ContextRetriever:
    def __init__(self, chunks: List[Document], embedder: SemanticEmbedder):
        self.index = SemanticIndex(chunks, embedder)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self.index.search(query, top_k)

    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Plain-text context for feeding the LLM prompts.

        The retrieved chunks are ordered by (page, chunk_id) rather than by
        score so that the text handed to FLAN-T5 reads in natural document
        order -- this is what actually matters for "generation done
        correctly after retrieval": scrambled, score-ordered context makes
        the model much more likely to generate a question/answer that
        doesn't track the source text. No metadata headers are injected
        into the prompt text itself; that would eat into the tight
        512-token input budget for no benefit to the model. Citation
        metadata is tracked separately (see _source_list).
        """
        if not results:
            return ""

        ordered = sorted(
            results,
            key=lambda item: (
                item["chunk"].metadata.get("page", 0),
                item["chunk"].metadata.get("chunk_id", 0),
            ),
        )

        return "\n\n".join(item["chunk"].page_content.strip() for item in ordered)


# ============================================================
# 9. FLAN-T5 GENERATOR
# ============================================================

class QuestionGenerator:
    def __init__(self, model_name: str = GENERATION_MODEL_NAME):
        print(f"[INFO] Loading generation model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        print("[INFO] Generation model loaded.")

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        deterministic: bool = False,
        min_new_tokens: Optional[int] = None,
        num_beams: int = 4,
        length_penalty: float = 1.0,
    ) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        gen_kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.15, no_repeat_ngram_size=3)

        if deterministic:
            # Beam search is far more consistent than sampling for grounded
            # extraction on a small (250M param) model. min_new_tokens stops
            # the model from stopping after a single short clause -- a
            # common flan-t5 failure mode -- and length_penalty > 1 biases
            # beam search toward the longer, more complete candidate instead
            # of the shortest one that satisfies the beams.
            gen_kwargs.update(do_sample=False, num_beams=num_beams, length_penalty=length_penalty)
            if min_new_tokens:
                gen_kwargs["min_new_tokens"] = min_new_tokens
        else:
            # Keep light sampling only for open-ended question phrasing,
            # where some variety is desirable, but tone it down from 0.7.
            gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.9)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------
    def generate_question(self, context: str, difficulty: str) -> str:
        prompt = f"""Generate ONE question at {difficulty} difficulty that requires a full, explanatory answer (not a one-word answer).

Use only the information in the reference text below. The question must:
- test an important concept from the text
- require explaining or describing something, not just naming a fact
- be answerable using only the reference text
- not contain the answer itself

Reference text:
{context}

Question:"""
        return self.generate_text(prompt, MAX_QUESTION_TOKENS, deterministic=False)

    # ------------------------------------------------------
    def generate_answer(self, question: str, context: str) -> str:
        # This is the dedicated RAG generation step: it only ever sees the
        # question plus the passages retrieved specifically for it, and is
        # explicitly instructed to write a complete, multi-sentence answer
        # rather than a bare fact fragment.
        prompt = f"""Answer the question fully and accurately using ONLY the reference text below.

Question: {question}

Reference text:
{context}

Instructions:
- Write a complete answer of 2 to 5 sentences.
- Explain the reasoning or supporting details from the reference text, not just a single word or phrase.
- Do not add information that is not in the reference text.
- Do not repeat the question in your answer.

Answer:"""
        return self.generate_text(
            prompt,
            MAX_ANSWER_TOKENS,
            deterministic=True,
            min_new_tokens=MIN_ANSWER_TOKENS,
            num_beams=4,
            length_penalty=1.4,
        )


# ============================================================
# 10. QUESTION CLEANING
# ============================================================

def clean_question(raw: str) -> str:
    if not raw:
        return ""

    raw = raw.strip()
    raw = re.sub(r"[*_`#]+", "", raw)
    raw = re.sub(r"^\s*question\s*:\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^\s*(?:Q\s*)?\d+[\.\):\-]\s*", "", raw, flags=re.IGNORECASE)

    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    for line in lines:
        if "?" in line:
            return line[: line.find("?") + 1].strip()

    if lines:
        question = lines[0].strip()
        if question and not question.endswith("?"):
            question += "?"
        return question

    return ""


# ============================================================
# 11. ANSWER CLEANING
# ============================================================

def clean_answer(raw: str) -> str:
    """
    Strip common generation artifacts from a short-answer response:
    leading "Answer:" echoes, quote wrapping, and markdown emphasis.
    Guards against accepting a degenerate/empty answer as valid.
    """
    if not raw:
        return ""

    text = raw.strip()
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip("\"'").strip()
    text = re.sub(r"\s+", " ", text)

    return text


def is_degenerate_answer(answer: str, question: str) -> bool:
    """
    Reject answers that are empty, just repeat the question back, are too
    short to be a real explanatory answer, or are implausibly long (a sign
    generation looped or dumped the whole reference text verbatim instead
    of composing an answer).
    """
    if not answer:
        return True

    normalized_answer = re.sub(r"\W+", " ", answer.lower()).strip()
    normalized_question = re.sub(r"\W+", " ", question.lower()).strip()

    if normalized_answer == normalized_question:
        return True

    word_count = len(answer.split())

    # Too short to be the multi-sentence explanatory answer we asked for.
    if word_count < 6:
        return True

    # Runaway generation / context dump rather than a composed answer.
    if word_count > 180:
        return True

    return False


# ============================================================
# 12. SYSTEM-WIDE RAG EVALUATION
# ============================================================

def evaluate_rag(qa_records: List[Dict[str, Any]], embedder: SemanticEmbedder) -> Dict[str, Any]:
    """
    Reference-free RAG evaluation over the WHOLE generated set, not just a
    single question. There's no ground-truth answer key to compare against,
    so this scores each question/answer/context triple using embedding
    cosine similarity as a proxy for three standard RAG quality axes:

    - context_relevance:  question <-> retrieved context
                           (did retrieval fetch material relevant to the
                           question at all?)
    - faithfulness:       answer <-> retrieved context
                           (is the answer actually grounded in what was
                           retrieved, rather than drifting from it?)
    - answer_relevance:   answer <-> question
                           (does the answer actually address the question
                           that was asked?)

    These are heuristic proxies (cosine similarity is not the same as a
    factual entailment check), but computed consistently across every
    question they give a single comparable system-level score, which is
    what's useful for judging "is this PDF's generated set good" rather
    than eyeballing each question individually.
    """
    if not qa_records:
        return {
            "per_question": [],
            "overall": {
                "context_relevance": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "rag_score": 0.0,
            },
            "note": "No questions were generated, so no evaluation could be computed.",
        }

    questions = [record["question"] for record in qa_records]
    answers = [record["answer"] for record in qa_records]
    contexts = [record["context"] for record in qa_records]

    # Embeddings are L2-normalized (see SemanticEmbedder.encode), so the
    # dot product below is already cosine similarity.
    question_embeddings = embedder.encode(questions)
    answer_embeddings = embedder.encode(answers)
    context_embeddings = embedder.encode(contexts)

    context_relevance = torch.sum(question_embeddings * context_embeddings, dim=1)
    faithfulness = torch.sum(answer_embeddings * context_embeddings, dim=1)
    answer_relevance = torch.sum(answer_embeddings * question_embeddings, dim=1)

    per_question = []
    for i, record in enumerate(qa_records):
        per_question.append(
            {
                "question": record["question"],
                "context_relevance": round(float(context_relevance[i]), 4),
                "faithfulness": round(float(faithfulness[i]), 4),
                "answer_relevance": round(float(answer_relevance[i]), 4),
            }
        )

    overall_context_relevance = float(context_relevance.mean())
    overall_faithfulness = float(faithfulness.mean())
    overall_answer_relevance = float(answer_relevance.mean())
    overall_rag_score = (overall_context_relevance + overall_faithfulness + overall_answer_relevance) / 3

    return {
        "per_question": per_question,
        "overall": {
            "context_relevance": round(overall_context_relevance, 4),
            "faithfulness": round(overall_faithfulness, 4),
            "answer_relevance": round(overall_answer_relevance, 4),
            "rag_score": round(overall_rag_score, 4),
        },
    }


# ============================================================
# 12b. GROUNDED ANSWER GRADING
# ============================================================

class NLIModel:
    """
    Natural-language-inference cross-encoder. Given a (premise, hypothesis)
    pair it reads BOTH texts together and returns the probability that the
    premise entails, contradicts, or is neutral to the hypothesis. Because it
    attends across the pair, it sees negation, antonyms and swapped roles
    that a bi-encoder cosine score cannot ("uses labeled data" vs "uses
    unlabeled data" are near-identical embeddings but a clear contradiction).
    """

    def __init__(self, candidates: List[str] = NLI_MODEL_CANDIDATES):
        errors = []
        for name in candidates:
            try:
                logger.info("Loading NLI cross-encoder: %s", name)
                print(f"[INFO] Loading NLI cross-encoder: {name}")
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE)
                model.eval()
                label_index = self._map_labels(model.config.id2label)
            except Exception as exc:  # try the next candidate
                errors.append(f"{name}: {exc}")
                logger.warning("NLI model %s unavailable: %s", name, exc)
                continue
            self.name, self.tokenizer, self.model, self.label_index = name, tokenizer, model, label_index
            print(f"[INFO] NLI cross-encoder loaded: {name}")
            break
        else:
            raise RuntimeError(
                "Could not load any NLI cross-encoder model. "
                "For the default model run `pip install sentencepiece`. Details: " + " | ".join(errors)
            )

    @staticmethod
    def _map_labels(id2label: Dict[Any, str]) -> Dict[str, int]:
        # Read the label order from the model's own config instead of
        # hardcoding it -- different NLI checkpoints order them differently.
        mapping = {}
        for index, label in id2label.items():
            lowered = str(label).lower()
            for key, needle in (("entailment", "entail"), ("contradiction", "contradict"), ("neutral", "neutral")):
                if needle in lowered:
                    mapping[key] = int(index)
        if set(mapping) != {"entailment", "contradiction", "neutral"}:
            raise RuntimeError(f"Unexpected NLI label set: {id2label}")
        return mapping

    def predict(self, pairs: List[Any]) -> List[Dict[str, float]]:
        """pairs: [(premise, hypothesis), ...] -> [{entailment, contradiction, neutral}, ...]"""
        results: List[Dict[str, float]] = []
        if not pairs:
            return results
        with torch.no_grad():
            for start in range(0, len(pairs), NLI_BATCH_SIZE):
                batch = pairs[start:start + NLI_BATCH_SIZE]
                encoded = self.tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    padding=True,
                    truncation=True,
                    max_length=NLI_MAX_TOKENS,
                    return_tensors="pt",
                )
                encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
                probabilities = torch.softmax(self.model(**encoded).logits, dim=-1).cpu().numpy()
                for row in probabilities:
                    results.append({name: float(row[index]) for name, index in self.label_index.items()})
        return results


class AnswerGrader:
    """
    Grades a student's answer against the SOURCE TEXT the question came from,
    not just against the generated reference answer (which can be incomplete
    or wrong).

    1. Key points. The reference answer is split into sentences; each one is
       kept as a graded "key point" only if the source passage entails it.
       Unsupported (hallucinated) reference sentences are dropped. This is
       computed once per question and cached.
    2. Coverage. For each key point: does the student's answer entail it?
       Full entailment = full credit; a close paraphrase that doesn't fully
       entail it earns partial credit from embedding similarity; if the
       student's text contradicts it, it earns nothing.
    3. Support. Each on-topic student sentence is checked against the most
       relevant source sentences (retrieve with embeddings, verify with NLI),
       so correct statements the reference omitted are not treated as errors
       and statements the source contradicts are caught.
    4. Grade = coverage x (1 - SUPPORT_WEIGHT + SUPPORT_WEIGHT x support),
       scaled down hard if anything the student wrote is contradicted.
    """

    def __init__(self, embedder: "SemanticEmbedder"):
        self.embedder = embedder
        self.nli = NLIModel()
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._profiles_lock = Lock()

    # ---------------- small helpers ----------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        embeddings = self.embedder.encode(texts)
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)

    def _nli(self, pairs: List[Any], memo: Dict[Any, Dict[str, float]]) -> List[Dict[str, float]]:
        """Batched NLI with a memo, so an identical (premise, hypothesis) pair is only ever scored once."""
        todo = [pair for pair in dict.fromkeys(pairs) if pair not in memo]
        for pair, probabilities in zip(todo, self.nli.predict(todo)):
            memo[pair] = probabilities
        return [memo[pair] for pair in pairs]

    @staticmethod
    def _split(text: str, cap: int) -> List[str]:
        parts: List[str] = []
        for sentence in split_into_sentences(text or ""):
            parts.extend(piece.strip() for piece in re.split(r"\s*;\s*|\n+", sentence) if piece.strip())
        parts = [piece for piece in parts if len(piece.split()) >= 2] or parts
        return parts[:cap]

    @staticmethod
    def _premises(sentences: List[str], similarities: np.ndarray, k: int) -> List[Any]:
        """
        The k sentences most similar to a claim, offered as individual premises
        plus (when k > 1) their concatenation in document order -- so a claim
        that needs two neighbouring source sentences can still be verified.
        Returns [(premise_text, relevance_similarity), ...].
        """
        k = min(k, len(sentences))
        top = [int(i) for i in np.argsort(-similarities)[:k]]
        premises = [(sentences[i], float(similarities[i])) for i in top]
        if k > 1:
            joined = " ".join(sentences[i] for i in sorted(top))
            premises.append((joined, float(similarities[top[0]])))
        return premises

    def _dedupe(self, sentences: List[str]) -> List[str]:
        if len(sentences) < 2:
            return sentences
        embeddings = self._embed(sentences)
        kept: List[int] = []
        for i in range(len(sentences)):
            if all(float(embeddings[i] @ embeddings[j]) < KEYPOINT_DEDUP_SIM for j in kept):
                kept.append(i)
        return [sentences[i] for i in kept]

    @staticmethod
    def _empty_result(note: str) -> Dict[str, Any]:
        return {
            "similarity": 0.0,
            "score": 0,
            "correct": False,
            "threshold": GRADE_CORRECT_THRESHOLD,
            "coverage": 0.0,
            "support": 0.0,
            "contradiction": 0.0,
            "embedding_similarity": 0.0,
            "grounded": False,
            "key_points": [],
            "missing_points": [],
            "contradicted_claims": [],
            "unsupported_claims": [],
            "ungrounded_reference_points": [],
            "note": note,
        }

    # ---------------- per-question profile (cached) ----------------

    def _profile(self, question: str, reference: str, context: str) -> Dict[str, Any]:
        key = hashlib.sha1("\x1f".join((question, reference, context)).encode("utf-8")).hexdigest()
        with self._profiles_lock:
            cached = self._profiles.get(key)
        if cached is not None:
            return cached
        profile = self._build_profile(question, reference, context)
        with self._profiles_lock:
            if len(self._profiles) >= PROFILE_CACHE_SIZE:
                self._profiles.pop(next(iter(self._profiles)))
            self._profiles[key] = profile
        return profile

    def _build_profile(self, question: str, reference: str, context: str) -> Dict[str, Any]:
        context_sentences = self._split(context, MAX_CONTEXT_SENTENCES)
        reference_sentences = self._split(reference, MAX_KEY_POINTS) or ([reference] if reference else [])
        grounded = bool(context_sentences)
        dropped: List[Dict[str, Any]] = []
        fallback_used = False

        if grounded:
            context_embeddings = self._embed(context_sentences)
            kept: List[str] = []
            if reference_sentences:
                reference_embeddings = self._embed(reference_sentences)
                similarities = reference_embeddings @ context_embeddings.T
                pairs, owners = [], []
                for r, sentence in enumerate(reference_sentences):
                    for premise, _ in self._premises(context_sentences, similarities[r], EVIDENCE_TOP_K):
                        pairs.append((premise, sentence))
                        owners.append(r)
                best_support = [0.0] * len(reference_sentences)
                for owner, probabilities in zip(owners, self._nli(pairs, {})):
                    best_support[owner] = max(best_support[owner], probabilities["entailment"])
                for sentence, support in zip(reference_sentences, best_support):
                    if support >= KEYPOINT_GROUNDING_MIN:
                        kept.append(sentence)
                    else:
                        dropped.append({"text": sentence, "support": round(support, 3)})

            if not kept:
                # Nothing in the reference is backed by the source (or there is
                # no reference): fall back to the source sentences that best
                # answer the question, since the source is the ground truth.
                fallback_used = True
                question_embedding = self._embed([question or reference])[0]
                scores = context_embeddings @ question_embedding
                top = sorted(int(i) for i in np.argsort(-scores)[:2])
                kept = [context_sentences[i] for i in top]
            evidence, evidence_embeddings = context_sentences, context_embeddings
        else:
            # Older material saved without its source passage: grade against the
            # reference answer alone (still semantic, just not source-grounded).
            kept = reference_sentences
            evidence = reference_sentences
            evidence_embeddings = self._embed(reference_sentences)

        key_points = self._dedupe(kept)
        profile = {
            "key_points": key_points,
            "key_point_embeddings": self._embed(key_points),
            "evidence": evidence,
            "evidence_embeddings": evidence_embeddings,
            "grounded": grounded,
            "fallback_used": fallback_used,
            "dropped": dropped,
        }
        logger.info(
            "grading profile built: grounded=%s key_points=%d dropped_unsupported=%d fallback=%s",
            grounded, len(key_points), len(dropped), fallback_used,
        )
        return profile

    # ---------------- grading ----------------

    def grade(self, question: str, reference_answer: str, student_answer: str, context: str = "") -> Dict[str, Any]:
        question = (question or "").strip()
        reference = re.sub(r"\s+", " ", reference_answer or "").strip()
        context = (context or "").strip()
        student = re.sub(r"\s+", " ", student_answer or "").strip()[:MAX_STUDENT_CHARS]

        if not student:
            return self._empty_result("No answer was given.")

        profile = self._profile(question, reference, context)
        key_points = profile["key_points"]
        if not key_points:
            return self._empty_result("No reference or source text was available to grade against.")

        claims = self._split(student, MAX_STUDENT_CLAIMS) or [student]
        claim_embeddings = self._embed(claims)
        whole_embedding = claim_embeddings if len(claims) == 1 else self._embed([student])
        question_embedding = self._embed([question])[0] if question else None
        reference_embedding = self._embed([reference])[0] if reference else None
        embedding_similarity = float(whole_embedding[0] @ reference_embedding) if reference_embedding is not None else 0.0

        if (
            question_embedding is not None
            and float(whole_embedding[0] @ question_embedding) >= ECHO_SIMILARITY
            and len(student.split()) <= len(question.split()) + 3
        ):
            result = self._empty_result("The answer only restates the question.")
            result["embedding_similarity"] = round(embedding_similarity, 4)
            return result

        memo: Dict[Any, Dict[str, float]] = {}

        # ---- coverage: is each key point entailed by what the student wrote? ----
        premise_texts = list(claims)
        premise_embeddings = claim_embeddings
        whole_index = None
        if len(claims) > 1:
            # Also try the whole answer as one premise, since a key point can be
            # spread across several of the student's sentences.
            premise_texts.append(student)
            premise_embeddings = np.vstack([claim_embeddings, whole_embedding])
            whole_index = len(premise_texts) - 1

        key_point_embeddings = profile["key_point_embeddings"]
        premise_vs_point = premise_embeddings @ key_point_embeddings.T  # [premises, key points]

        pairs, owners = [], []
        for k, point in enumerate(key_points):
            for p, premise in enumerate(premise_texts):
                if p == whole_index or premise_vs_point[p, k] >= NLI_MIN_RELEVANCE:
                    pairs.append((premise, point))
                    owners.append((k, p))

        entailment = [0.0] * len(key_points)
        point_contradiction = [0.0] * len(key_points)
        for (k, p), probabilities in zip(owners, self._nli(pairs, memo)):
            entailment[k] = max(entailment[k], probabilities["entailment"])
            if premise_vs_point[p, k] >= CONTRADICTION_MIN_RELEVANCE:
                point_contradiction[k] = max(point_contradiction[k], probabilities["contradiction"])

        for k in range(len(key_points)):
            if entailment[k] >= ENTAILMENT_OVERRIDE:
                point_contradiction[k] = 0.0

        closeness = np.clip(
            (premise_vs_point.max(axis=0) - SOFT_SIM_FLOOR) / (SOFT_SIM_CEIL - SOFT_SIM_FLOOR), 0.0, 1.0
        )
        point_coverage = []
        for k in range(len(key_points)):
            if point_contradiction[k] >= CONTRADICTION_THRESHOLD:
                point_coverage.append(0.0)
            else:
                point_coverage.append(max(entailment[k], SOFT_CREDIT_WEIGHT * float(closeness[k])))

        # ---- support: is each on-topic claim backed by the source? ----
        evidence = profile["evidence"]
        evidence_embeddings = profile["evidence_embeddings"]
        claim_vs_evidence = claim_embeddings @ evidence_embeddings.T
        claim_vs_question = claim_embeddings @ question_embedding if question_embedding is not None else np.zeros(len(claims))

        pairs, owners, relevant = [], [], []
        for c, claim in enumerate(claims):
            if max(float(claim_vs_evidence[c].max()), float(claim_vs_question[c])) < CLAIM_RELEVANCE_MIN:
                continue  # filler: neither rewarded nor penalised
            relevant.append(c)
            for premise, relevance in self._premises(evidence, claim_vs_evidence[c], EVIDENCE_TOP_K):
                pairs.append((premise, claim))
                owners.append((c, relevance))

        claim_support = {c: 0.0 for c in relevant}
        claim_contradiction = {c: 0.0 for c in relevant}
        for (c, relevance), probabilities in zip(owners, self._nli(pairs, memo)):
            claim_support[c] = max(claim_support[c], probabilities["entailment"])
            if relevance >= CONTRADICTION_MIN_RELEVANCE:
                claim_contradiction[c] = max(claim_contradiction[c], probabilities["contradiction"])

        for c in relevant:
            if claim_support[c] >= ENTAILMENT_OVERRIDE:
                claim_contradiction[c] = 0.0

        # ---- combine ----
        coverage = float(np.mean(point_coverage))
        support = float(np.mean(list(claim_support.values()))) if claim_support else 0.0
        contradiction = max(point_contradiction + list(claim_contradiction.values()) + [0.0])
        contradicted = contradiction >= CONTRADICTION_THRESHOLD
        penalty_factor = (1.0 - CONTRADICTION_PENALTY * contradiction) if contradicted else 1.0
        quality = coverage * ((1.0 - SUPPORT_WEIGHT) + SUPPORT_WEIGHT * support) * penalty_factor
        quality = float(min(1.0, max(0.0, quality)))

        return {
            # Kept under the original keys so every existing page keeps working.
            # `similarity` is now the grounded grade (0-1), so score == round(similarity * 10)
            # and correct == (similarity >= threshold), exactly as before.
            "similarity": round(quality, 4),
            "score": int(round(quality * 10)),
            "correct": quality >= GRADE_CORRECT_THRESHOLD,
            "threshold": GRADE_CORRECT_THRESHOLD,
            # New, explainable components.
            "coverage": round(coverage, 4),
            "support": round(support, 4),
            "contradiction": round(contradiction, 4),
            "embedding_similarity": round(embedding_similarity, 4),
            "grounded": profile["grounded"],
            "key_points": [
                {
                    "text": point,
                    "coverage": round(point_coverage[k], 3),
                    "entailment": round(entailment[k], 3),
                    "contradicted": point_contradiction[k] >= CONTRADICTION_THRESHOLD,
                }
                for k, point in enumerate(key_points)
            ],
            "missing_points": [point for k, point in enumerate(key_points) if point_coverage[k] < 0.5],
            "contradicted_claims": [claims[c] for c, v in claim_contradiction.items() if v >= CONTRADICTION_THRESHOLD],
            "unsupported_claims": [
                claims[c] for c, v in claim_support.items()
                if v < 0.30 and claim_contradiction[c] < CONTRADICTION_THRESHOLD
            ],
            "ungrounded_reference_points": profile["dropped"],
            "note": "",
        }


# ============================================================
# 13. PIPELINE
# ============================================================

class QuestionGeneratorPipeline:
    def __init__(self):
        print("\n==========================================")
        print(" Initializing NLP Question Generator")
        print("==========================================")

        self.embedder = SemanticEmbedder()
        self.chunker = SemanticChunker(self.embedder)
        self.generator = QuestionGenerator()
        self.grader = AnswerGrader(self.embedder)

        print("==========================================")
        print(" Pipeline ready")
        print("==========================================\n")

    def validate_input(self, difficulty: str, num_questions: int):
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Invalid difficulty. Choose from {sorted(DIFFICULTIES)}")
        if num_questions < 1:
            raise ValueError("num_questions must be at least 1.")
        if num_questions > 50:
            raise ValueError("Maximum 50 questions allowed.")

    def _source_list(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "page": item["chunk"].metadata.get("page"),
                "section": item["chunk"].metadata.get("section"),
                "chunk_id": item["chunk"].metadata.get("chunk_id"),
                "similarity": round(item["score"], 4),
            }
            for item in results
        ]

    def _build_seed_pool(
        self, retriever: "ContextRetriever", chunks: List[Document], topic: Optional[str], num_questions: int
    ) -> List[Document]:
        """
        Produces the ordered list of "seed" chunks that question generation
        rotates through -- one different seed per generation attempt -- so
        that questions are actually drawn from different parts of the
        document instead of the same handful of chunks every time.

        Previously every question (topic or not) was generated from one
        fixed block of context: with a topic, the same top-K retrieval
        result reused for every attempt; without a topic, literally
        chunks[:TOPIC_TOP_K] -- the first few chunks of page 1 -- for the
        entire run. Both guarantee near-duplicate questions once you ask
        for more than a couple.
        """
        # Administrative/front-matter chunks (course codes, instructor
        # lines, syllabus boilerplate, ...) are excluded from the seed
        # candidates -- they still exist in the retrieval index and can
        # show up as supporting context, but a question is never generated
        # FROM one of them directly.
        seedable_chunks = [c for c in chunks if not is_boilerplate_chunk(c.page_content)]
        if not seedable_chunks:
            seedable_chunks = list(chunks)

        pool_size = min(len(seedable_chunks), max(TOPIC_TOP_K, num_questions * 3))

        if topic:
            topic_query = f"Important educational concepts about {topic}"
            # Over-retrieve against the full index, then drop boilerplate
            # hits and trim back to pool_size, so filtering doesn't starve
            # the pool when the top raw hits happen to be admin text.
            results = retriever.retrieve(topic_query, pool_size * 2)
            results = [item for item in results if not is_boilerplate_chunk(item["chunk"].page_content)][:pool_size]
            # Keep retrieval-relevance order but topic-relevant material can
            # still cluster (many top hits from the same paragraph) -- a
            # light interleave by document position spreads them out while
            # still respecting "relevant to the topic".
            results = sorted(results, key=lambda item: item["chunk"].metadata.get("chunk_id", 0))
            seeds = [item["chunk"] for item in results]
        else:
            # No topic: spread seeds evenly across the WHOLE document
            # instead of only ever using the first few chunks.
            if pool_size >= len(seedable_chunks):
                seeds = list(seedable_chunks)
            else:
                stride = len(seedable_chunks) / pool_size
                seeds = [seedable_chunks[int(i * stride)] for i in range(pool_size)]

        return seeds or list(chunks[:1])

    def _seed_context(self, chunks: List[Document], seed_chunk: Document) -> str:
        """Local context window around a seed chunk (by chunk_id, which is
        the same as its index in `chunks` since ids are assigned in order).
        Boilerplate neighbors are dropped from the window rather than
        included, so admin text can't leak into the question prompt just
        because it happened to sit next to a good chunk."""
        seed_id = seed_chunk.metadata.get("chunk_id", 0)
        start = max(0, seed_id - QUESTION_CONTEXT_WINDOW)
        end = min(len(chunks), seed_id + QUESTION_CONTEXT_WINDOW + 1)
        window = [c for c in chunks[start:end] if c is seed_chunk or not is_boilerplate_chunk(c.page_content)]
        return "\n\n".join(c.page_content.strip() for c in window)


    def _generate_rag_answer(self, retriever: "ContextRetriever", question: str) -> Optional[Dict[str, Any]]:
        """
        The dedicated answer RAG step. Takes the already-generated question
        as a fresh query, retrieves the passages most relevant to THAT
        question specifically (independent of whatever broader context the
        question itself was generated from), and generates the answer
        grounded only in that retrieved context.

        This is a distinct retrieval + generation pass, not a reuse of the
        question-generation context -- retrieval and generation are two
        separate stages here, which is what makes it an actual RAG pipeline
        rather than "generate everything from one blob of context".

        Retries generation a few times against the SAME retrieved context
        before giving up on the question entirely.
        """
        results = retriever.retrieve(question, ANSWER_TOP_K)
        if not results:
            return None

        context = retriever.build_context(results)

        # Answerability gate: if the question's own wording doesn't land
        # close to anything actually in the document, it's very likely
        # incoherent/malformed (fused from unrelated sentences) rather than
        # a real question about this material, however grammatical it
        # looks. Checked against the best-matching CONTEXT SENTENCE (not a
        # whole-passage pooled embedding) so a long multi-chunk context
        # doesn't dilute a genuinely good match. Caught here, before
        # spending a generation call on an answer that would just be
        # ungrounded.
        context_sentences = split_into_sentences(context) or [context]
        context_sentence_embeddings = self.embedder.encode(context_sentences)
        question_embedding = self.embedder.encode([question])[0]
        best_context_relevance = float(torch.matmul(context_sentence_embeddings, question_embedding).max())
        if best_context_relevance < MIN_QUESTION_CONTEXT_RELEVANCE:
            print(
                f"[WARN] Question failed answerability gate "
                f"(best_context_relevance={best_context_relevance:.4f} < {MIN_QUESTION_CONTEXT_RELEVANCE}): {question}"
            )
            return None

        for attempt in range(1, ANSWER_RETRIES_PER_QUESTION + 1):
            raw_answer = self.generator.generate_answer(question, context)
            print(f"[DEBUG] Raw answer (attempt {attempt}): {raw_answer}")

            answer = clean_answer(raw_answer)
            if is_degenerate_answer(answer, question):
                print("[INFO] Degenerate answer. Retrying.")
                continue

            return {
                "answer": answer,
                "context": context,
                "source": self._source_list(results),
            }

        return None

    def generate(
        self,
        pdf_path: str,
        topic: Optional[str] = None,
        difficulty: str = "medium",
        num_questions: int = 5,
        progress_callback=None,
    ) -> Dict[str, Any]:

        logger.info(
            "QuestionGeneratorPipeline.generate: start topic=%r difficulty=%s num_questions=%d",
            topic,
            difficulty,
            num_questions,
        )

        def report(stage):
            if progress_callback is not None:
                progress_callback(stage)

        self.validate_input(difficulty, num_questions)

        report("cleaning")
        documents = clean_documents(load_pdf(pdf_path))
        report("chunking")
        chunks = self.chunker.split(documents)
        if not chunks:
            raise ValueError("No semantic chunks were created.")

        report("embedding")
        retriever = ContextRetriever(chunks, self.embedder)

        if topic:
            topic_query = f"Important educational concepts about {topic}"
            topic_results = retriever.retrieve(topic_query, TOPIC_TOP_K)
        else:
            topic_results = [{"chunk": chunk, "score": 1.0} for chunk in chunks[:TOPIC_TOP_K]]

        if not topic_results:
            raise ValueError("No relevant context was found.")

        # Seeds rotate through different parts of the document so each
        # generation attempt starts from different material, instead of
        # every question being generated from the same fixed context block.
        seed_pool = self._build_seed_pool(retriever, chunks, topic, num_questions)

        final_questions = []
        report("generating")
        generated_question_keys = set()
        accepted_question_embeddings: List[torch.Tensor] = []
        attempts = 0
        # Retries now happen per-question inside _generate_rag_answer, so
        # the outer attempt budget can stay modest -- it just needs enough
        # room to replace a question that never produced a usable answer.
        max_attempts = num_questions * 6

        while len(final_questions) < num_questions and attempts < max_attempts:
            attempts += 1
            print(f"\n[INFO] Generating question {len(final_questions) + 1}/{num_questions} | attempt {attempts}")

            # Every attempt (not just every successful question) advances to
            # the next seed, so a rejected/duplicate attempt also moves on
            # to different material rather than re-rolling the same context.
            seed_chunk = seed_pool[(attempts - 1) % len(seed_pool)]
            generation_context = self._seed_context(chunks, seed_chunk)

            raw_question = self.generator.generate_question(generation_context, difficulty)
            print(f"[DEBUG] Raw question:\n{raw_question}")

            question = clean_question(raw_question)
            if not question:
                print("[WARN] Empty question. Retrying.")
                continue

            question_key = re.sub(r"\W+", " ", question.lower()).strip()
            if question_key in generated_question_keys:
                print("[WARN] Duplicate question (exact match).")
                continue

            # Exact-string dedup misses paraphrases ("...basic layer?" vs
            # "...basic layer called?" vs "...basic structure?"), which is
            # exactly what was slipping through before. Catch those via
            # embedding similarity against every question accepted so far.
            question_embedding = self.embedder.encode([question])[0]
            if accepted_question_embeddings:
                sims = torch.stack(accepted_question_embeddings) @ question_embedding
                if float(sims.max()) >= QUESTION_DEDUP_THRESHOLD:
                    print(f"[WARN] Duplicate question (semantic match, sim={float(sims.max()):.4f}).")
                    continue

            generated_question_keys.add(question_key)

            # Dedicated RAG step: the question feeds a fresh retrieval
            # (independent of the seed context it was generated from), and
            # the answer is generated only from what that retrieval returns.
            rag_result = self._generate_rag_answer(retriever, question)
            if rag_result is None:
                print("[WARN] Could not produce a valid answer for this question.")
                continue

            accepted_question_embeddings.append(question_embedding)

            final_questions.append(
                {
                    "question": question,
                    "answer": rag_result["answer"],
                    "type": QUESTION_TYPE,
                    "difficulty": difficulty,
                    "context": rag_result["context"],
                    "source": rag_result["source"],
                }
            )
            logger.info("QuestionGeneratorPipeline.generate: accepted question %d/%d", len(final_questions), num_questions)

        pages_used = sorted(
            {
                item["chunk"].metadata.get("page")
                for item in topic_results
                if item["chunk"].metadata.get("page") is not None
            }
        )

        evaluation = evaluate_rag(final_questions, self.embedder)

        print("\n[INFO] RAG evaluation (system-wide):")
        print(f"       context_relevance : {evaluation['overall']['context_relevance']}")
        print(f"       faithfulness      : {evaluation['overall']['faithfulness']}")
        print(f"       answer_relevance  : {evaluation['overall']['answer_relevance']}")
        print(f"       rag_score         : {evaluation['overall']['rag_score']}")

        return {
            "questions": final_questions,
            "type": QUESTION_TYPE,
            "difficulty": difficulty,
            "requested_questions": num_questions,
            "generated_questions": len(final_questions),
            "topic": topic,
            "pages_used": pages_used,
            "semantic_chunks_created": len(chunks),
            "retrieval_chunks_used": len(topic_results),
            "retrieval": self._source_list(topic_results),
            "evaluation": evaluation,
        }
        logger.info(
            "QuestionGeneratorPipeline.generate: complete generated=%d pages=%s chunks=%d",
            len(final_questions),
            pages_used,
            len(chunks),
        )
        return response


# ============================================================
# GLOBAL PIPELINE
# ============================================================

pipeline = QuestionGeneratorPipeline()


# ============================================================
# PUBLIC API
# ============================================================

def generate_questions(
    pdf_path: str,
    topic: Optional[str] = None,
    difficulty: str = "medium",
    num_questions: int = 5,
    progress_callback=None,
) -> Dict[str, Any]:
    logger.info("generate_questions: called pdf_path=%s", pdf_path)
    return pipeline.generate(
        pdf_path=pdf_path,
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
    )


def grade_answer(
    question: str,
    reference_answer: str,
    student_answer: str,
    context: str = "",
) -> Dict[str, Any]:
    """Grade one student answer, grounded in the question's source passage (`context`)."""
    return pipeline.grader.grade(question, reference_answer, student_answer, context)