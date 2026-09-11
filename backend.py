# ============================================================
# backend.py
#
# PDF -> page-aware extraction -> cleaning -> semantic chunking
# -> embeddings -> topic retrieval -> question generation
# -> question-specific retrieval -> short-answer generation
# -> final question
#
# NOTE: MCQ/option generation was removed. Generating four
# plausible distractors per question meant an extra heavy
# seq2seq call, plus an embedding-based dedup pass, for every
# single question -- by far the most CPU-expensive part of the
# pipeline. This build only ever produces short-answer Q&A
# pairs, which need exactly one generation call per question.
# ============================================================

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
QUESTION_TOP_K = 4

# Generation
# NOTE: flan-t5 is fine-tuned mostly on <=512 token inputs. 768 was pushing
# the model outside its trained context and degrading output quality.
MAX_INPUT_TOKENS = 512
MAX_QUESTION_TOKENS = 64
MAX_ANSWER_TOKENS = 48

# How many times to retry answer generation for the SAME question before
# giving up on that question and generating a fresh one instead.
ANSWER_RETRIES_PER_QUESTION = 2

QUESTION_TYPE = "Short Answer"
DIFFICULTIES = {"easy", "medium", "hard"}

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
    ) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        gen_kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.15, no_repeat_ngram_size=3)

        if deterministic:
            # Structured, single-fact answers need reliable, literal
            # extraction rather than creative variation. Beam search is
            # far more consistent than sampling for this on a small
            # (250M param) model.
            gen_kwargs.update(do_sample=False, num_beams=2)
        else:
            # Keep light sampling only for open-ended question phrasing,
            # where some variety is desirable, but tone it down from 0.7.
            gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.9)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------
    def generate_question(self, context: str, difficulty: str) -> str:
        prompt = f"""Generate ONE short-answer question at {difficulty} difficulty.

Use only the information in the reference text below. The question must:
- test an important concept from the text
- be answerable in a short phrase or sentence using only the reference text
- not contain the answer itself

Reference text:
{context}

Question:"""
        return self.generate_text(prompt, MAX_QUESTION_TOKENS, deterministic=False)

    # ------------------------------------------------------
    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""Answer the question using only the reference text. Give the shortest correct answer possible, as a short phrase or sentence.

Question: {question}

Reference text:
{context}

Answer:"""
        return self.generate_text(prompt, MAX_ANSWER_TOKENS, deterministic=True)


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
    Reject answers that are empty, just repeat the question back, or are
    implausibly long for a "short answer" (a sign generation looped or
    copied the whole reference text instead of extracting a fact).
    """
    if not answer:
        return True

    normalized_answer = re.sub(r"\W+", " ", answer.lower()).strip()
    normalized_question = re.sub(r"\W+", " ", question.lower()).strip()

    if normalized_answer == normalized_question:
        return True

    if len(answer.split()) > 40:
        return True

    return False


# ============================================================
# 12. PIPELINE
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

    def _build_short_answer(self, question: str, question_context: str) -> Optional[str]:
        """
        Retries answer generation for the SAME question (using the SAME
        retrieved context) a few times before giving up on it, rather than
        discarding a perfectly good question after a single degenerate
        generation.
        """
        for attempt in range(1, ANSWER_RETRIES_PER_QUESTION + 1):
            raw_answer = self.generator.generate_answer(question, question_context)
            print(f"[DEBUG] Raw answer (attempt {attempt}): {raw_answer}")

            answer = clean_answer(raw_answer)
            if is_degenerate_answer(answer, question):
                print("[INFO] Degenerate answer. Retrying.")
                continue

            return answer

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

        topic_context = retriever.build_context(topic_results)

        final_questions = []
        report("generating")
        generated_question_keys = set()
        attempts = 0
        # Retries now happen per-question inside _build_short_answer, so
        # the outer attempt budget can stay modest -- it just needs enough
        # room to replace a question that never produced a usable answer.
        max_attempts = num_questions * 6

        while len(final_questions) < num_questions and attempts < max_attempts:
            attempts += 1
            print(f"\n[INFO] Generating question {len(final_questions) + 1}/{num_questions} | attempt {attempts}")

            raw_question = self.generator.generate_question(topic_context, difficulty)
            print(f"[DEBUG] Raw question:\n{raw_question}")

            question = clean_question(raw_question)
            if not question:
                print("[WARN] Empty question. Retrying.")
                continue

            question_key = re.sub(r"\W+", " ", question.lower()).strip()
            if question_key in generated_question_keys:
                print("[WARN] Duplicate question.")
                continue
            generated_question_keys.add(question_key)

            # Retrieval happens fresh, per-question, against the specific
            # question text -- not reused from the broader topic context --
            # so the answer generation step below is grounded in the chunks
            # most relevant to THIS question rather than the whole topic.
            question_results = retriever.retrieve(question, QUESTION_TOP_K)
            if not question_results:
                print("[WARN] No context for question.")
                continue

            question_context = retriever.build_context(question_results)

            answer = self._build_short_answer(question, question_context)
            if answer is None:
                print("[WARN] Could not produce a valid answer for this question.")
                continue

            final_questions.append(
                {
                    "question": question,
                    "answer": answer,
                    "type": QUESTION_TYPE,
                    "difficulty": difficulty,
                    "source": self._source_list(question_results),
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

        response = {
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
        progress_callback=progress_callback,
    )