# ============================================================
# backend.py
#
# PDF -> page-aware extraction -> cleaning -> semantic chunking
# -> embeddings -> topic retrieval -> question generation
# -> [RAG] question-specific retrieval -> long-form answer generation
# -> final question + system-wide RAG evaluation score
# ============================================================

import re
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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

QUESTION_TYPE = "Long Answer"
DIFFICULTIES = {"easy", "medium", "hard"}

print(f"[INFO] Device: {DEVICE}")


# ============================================================
# 1. PDF LOADING
# ============================================================

def load_pdf(file_path: str) -> List[Document]:
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
    result = []
    for document in documents:
        text = clean_text(document.page_content)
        if not text:
            continue
        result.append(Document(page_content=text, metadata=document.metadata))
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
        all_chunks = []
        chunk_id = 0
        for document in documents:
            for chunk in self.chunk_page(document):
                chunk.metadata["chunk_id"] = chunk_id
                all_chunks.append(chunk)
                chunk_id += 1

        print(f"[INFO] Semantic chunks: {len(all_chunks)}")
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
        pool_size = min(len(chunks), max(TOPIC_TOP_K, num_questions * 3))

        if topic:
            topic_query = f"Important educational concepts about {topic}"
            results = retriever.retrieve(topic_query, pool_size)
            # Keep retrieval-relevance order but topic-relevant material can
            # still cluster (many top hits from the same paragraph) -- a
            # light interleave by document position spreads them out while
            # still respecting "relevant to the topic".
            results = sorted(results, key=lambda item: item["chunk"].metadata.get("chunk_id", 0))
            seeds = [item["chunk"] for item in results]
        else:
            # No topic: spread seeds evenly across the WHOLE document
            # instead of only ever using the first few chunks.
            if pool_size >= len(chunks):
                seeds = list(chunks)
            else:
                stride = len(chunks) / pool_size
                seeds = [chunks[int(i * stride)] for i in range(pool_size)]

        return seeds or list(chunks[:1])

    def _seed_context(self, chunks: List[Document], seed_chunk: Document) -> str:
        """Local context window around a seed chunk (by chunk_id, which is
        the same as its index in `chunks` since ids are assigned in order)."""
        seed_id = seed_chunk.metadata.get("chunk_id", 0)
        start = max(0, seed_id - QUESTION_CONTEXT_WINDOW)
        end = min(len(chunks), seed_id + QUESTION_CONTEXT_WINDOW + 1)
        return "\n\n".join(c.page_content.strip() for c in chunks[start:end])


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
    ) -> Dict[str, Any]:

        self.validate_input(difficulty, num_questions)

        documents = clean_documents(load_pdf(pdf_path))
        chunks = self.chunker.split(documents)
        if not chunks:
            raise ValueError("No semantic chunks were created.")

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
) -> Dict[str, Any]:
    return pipeline.generate(
        pdf_path=pdf_path,
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
    )
