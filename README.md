# Question Lab — NLP Exam Generator & Grader

Turn any text-based PDF into a bank of explanatory questions with source-grounded reference answers, run them as an exam for approved students, and grade the written answers **by meaning, not by wording**.

Everything runs **locally** with open models — no LLM API, no external services. The interesting part of this project is the NLP: semantic chunking, retrieval-augmented question/answer generation, and a grading method built on **natural language inference (NLI)** and sentence embeddings, grounded in the original source text.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [NLP pipeline: PDF → questions](#3-nlp-pipeline-pdf--questions)
4. [NLP pipeline: grading answers](#4-nlp-pipeline-grading-answers)
5. [Design principles](#5-design-principles)
6. [Known limitations](#6-known-limitations)
7. [Getting started](#7-getting-started)
8. [Using the app](#8-using-the-app)
9. [API reference](#9-api-reference)
10. [Configuration reference](#10-configuration-reference)
11. [Data storage](#11-data-storage)
12. [Security notes](#12-security-notes)
13. [Troubleshooting](#13-troubleshooting)
14. [Ideas for future work](#14-ideas-for-future-work)

---

## 1. What it does

| Role | Can do |
|---|---|
| **Admin** (one account, username + password) | Upload PDFs · generate questions · select questions and publish them as an **exam** · approve student email addresses · review every submission with a per-question automated grade and explanation · grade a student's answers uploaded as a PDF · export questions to PDF |
| **Student** (approved email + own password) | Take open exams · practise in **Quiz Mode** with instant self-graded feedback |

End-to-end flow:

```
Admin uploads PDF ─► questions + reference answers generated (grounded in the PDF)
        │
        ▼
Admin picks questions ─► publishes an exam ─► approves student emails
        │
        ▼
Student logs in ─► answers in free text ─► submits
        │
        ▼
Answers graded automatically (NLI + embeddings, grounded in the source passage)
        │
        ▼
Admin reviews scores, covered/missing key points, contradictions
```

---

## 2. Architecture at a glance

```
┌────────────────────────── Browser (static HTML + JS pages) ─────────────────────────┐
│ index · materials · students · review · evaluate │ student · quiz │ login pages      │
└───────────────────────────────────────┬─────────────────────────────────────────────┘
                                        │ JSON over HTTP, signed-cookie sessions
┌───────────────────────────────────────▼─────────────────────────────────────────────┐
│ main.py  — FastAPI: auth, student accounts, jobs, exams, submissions, PDF export     │
│            JSON files in uploads/  (materials, exams, submissions, students)         │
└───────────────────────────────────────┬─────────────────────────────────────────────┘
                                        │ function calls
┌───────────────────────────────────────▼─────────────────────────────────────────────┐
│ backend.py — ALL the NLP                                                             │
│   SemanticEmbedder (MiniLM)  ·  SemanticChunker  ·  SemanticIndex / ContextRetriever │
│   QuestionGenerator (FLAN-T5) · QuestionGeneratorPipeline · evaluate_rag             │
│   NLIModel (cross-encoder)    · AnswerGrader  ·  grade_answer()                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Models

| Purpose | Model | Notes |
|---|---|---|
| Sentence embeddings — chunking, retrieval, de-duplication, answerability gate, soft similarity in grading | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, L2-normalised, so a dot product *is* cosine similarity. Small and fast on CPU. |
| Question & answer generation | `google/flan-t5-base` | ~250M-parameter instruction-tuned encoder–decoder. Input is limited to 512 tokens. |
| Natural-language inference for grading | `cross-encoder/nli-deberta-v3-small` (falls back to `cross-encoder/nli-MiniLM2-L6-H768`) | A cross-encoder reads premise **and** hypothesis together and returns P(entailment / contradiction / neutral). The label order is read from the model's own config, never hardcoded. |

All models are loaded **once at startup** (so startup is slow and uses a few GB of RAM) and run on GPU automatically if CUDA is available.

---

## 3. NLP pipeline: PDF → questions

Everything in this section lives in `backend.py`. Each stage below names the function/class and the constants that control it.

```
PDF ─► page text ─► clean ─► sentences ─► semantic chunks ─► embedding index
                                                │
              ┌─────────────────────────────────┘
              ▼
   seed pool (boilerplate removed, spread across the document)
              │   one seed per attempt
              ▼
   question generation (FLAN-T5, sampled)
              │
              ▼   quality gates: exact dup → semantic dup → answerability
              │
              ▼
   RAG answer: retrieve for the *question* ─► FLAN-T5 (beam search) ─► answer filter
              │
              ▼
   question + answer + source passage + citations  ─►  system RAG score
```

### 3.1 Ingestion and cleaning
`load_pdf` extracts text page by page with LangChain's `PyPDFLoader`, keeping the page number. Pages with no text are skipped; if nothing is readable the request fails with a clear error. `clean_text` collapses all whitespace and removes "Page N (of M)" markers.

> Text-based PDFs only — there is no OCR, so scanned documents yield nothing.

### 3.2 Sentence segmentation
`split_into_sentences` splits on `. ! ?` followed by whitespace, after protecting common abbreviations (`e.g`, `i.e`, `Mr`, `Mrs`, `Dr`, `etc`) so they don't end a sentence. It is deliberately simple and dependency-free.

### 3.3 Semantic chunking
Fixed-size chunks cut ideas in half. Instead, `SemanticChunker` **splits where the topic changes**:

1. Embed every sentence on a page.
2. Compute the cosine similarity between each sentence and the next.
3. Start a new chunk when similarity falls below **`SEMANTIC_SIMILARITY_THRESHOLD = 0.45`** (and the current chunk already has ≥ **`MIN_SENTENCES_PER_CHUNK = 2`**), or when the chunk reaches **`MAX_SENTENCES_PER_CHUNK = 6`**.

Chunks never cross page boundaries. Each chunk carries metadata: page, a heuristic `section` heading (numbered heading / "Chapter N" / ALL-CAPS line), `chunk_id` (global, in reading order), sentence count and length.

### 3.4 Dense retrieval index
`SemanticIndex` embeds every chunk once and keeps the matrix in memory. `search` embeds the query, takes dot products against all chunks and returns the top-k with scores (`torch.topk`). There is no vector database — at the size of a course document, brute-force matrix multiplication is exact and instantaneous.

`ContextRetriever.build_context` re-orders retrieved chunks **by document position (page, chunk_id), not by score**, so the generator reads the passage in natural order. No metadata headers are injected into the prompt text; they would only waste the 512-token budget. Citations are tracked separately.

### 3.5 Choosing what to ask about: the seed pool
Early versions generated every question from the same context block and produced near-identical questions. The generator now rotates through a **seed pool** — a different chunk for every attempt:

- **Boilerplate is excluded** (`is_boilerplate_chunk`): chunks matching administrative patterns (`course code`, `instructor`, `credits`, `ISBN`, `syllabus`, `copyright`, …) or shorter than **`MIN_SEEDABLE_WORDS = 12`** words can't be seeds. They stay in the index and may still appear as supporting context.
- **With a `topic`**: retrieve the best-matching chunks for *"Important educational concepts about {topic}"*, drop boilerplate, then sort by position so seeds are spread through the material.
- **Without a topic**: take evenly spaced chunks across the *whole* document.
- Pool size is `max(TOPIC_TOP_K, 3 × num_questions)`, capped by the number of eligible chunks.
- The prompt context for a seed is the seed plus its neighbours (`QUESTION_CONTEXT_WINDOW = 1` chunk each side), with boilerplate neighbours removed.

### 3.6 Question generation
`QuestionGenerator.generate_question` prompts FLAN-T5 to produce **one** question that needs an *explanatory* answer, uses only the supplied text, and doesn't contain its own answer. Decoding is **sampled** (temperature 0.6, top-p 0.9, repetition penalty 1.15, no-repeat 3-grams, ≤ 64 new tokens) so different seeds give varied wording. `clean_question` strips markdown, "Question:" prefixes and numbering, and keeps the first sentence ending in `?`.

### 3.7 Question quality control
A candidate question must pass three gates before any answer is attempted:

| Gate | What it catches | How |
|---|---|---|
| **Exact duplicate** | Verbatim repeats | Normalised string match against accepted questions |
| **Semantic duplicate** | Paraphrases ("…basic layer?" / "…basic layer called?") | Cosine similarity to every accepted question; reject at ≥ **`QUESTION_DEDUP_THRESHOLD = 0.85`** |
| **Answerability** | Incoherent questions fused from unrelated sentences | After retrieving context for the question, its best-matching **context sentence** must reach cosine ≥ **`MIN_QUESTION_CONTEXT_RELEVANCE = 0.30`**. A question that matches nothing real in the document is rejected before spending a generation call. |

Every *attempt* (accepted or not) advances to the next seed, and the loop runs at most `6 × num_questions` attempts.

### 3.8 Answer generation as RAG
Question and answer are produced in **separate stages** — the answer never sees the context the question was written from:

1. **Retrieve for the question** — fresh dense retrieval of the top **`ANSWER_TOP_K = 6`** chunks using the question as the query.
2. **Generate** — FLAN-T5 with **deterministic beam search** (4 beams, `length_penalty = 1.4`), `min_new_tokens = 24`, `max_new_tokens = 220`. Beam search is used because it follows format far more reliably than sampling on a small model; the minimum length and length penalty counter FLAN-T5's habit of stopping after one clause.
3. **Filter** (`is_degenerate_answer`) — reject empty answers, answers that merely repeat the question, answers under 6 words, and runaway output over 180 words. Up to **`ANSWER_RETRIES_PER_QUESTION = 2`** attempts reuse the same retrieved context before the question is abandoned.
4. **Store** — each accepted item keeps the question, answer, difficulty, **the retrieved source passage (`context`)** and citation metadata (page, section, chunk id, similarity). The stored passage is what later grounds the grading.

### 3.9 System-level RAG score (a monitoring signal)
After generation, `evaluate_rag` produces one reference-free score for the whole set from embedding cosine similarities:

| Metric | Compares |
|---|---|
| `context_relevance` | question ↔ retrieved context |
| `faithfulness` | answer ↔ retrieved context |
| `answer_relevance` | answer ↔ question |
| `rag_score` | mean of the three |

It is printed to the server log and returned as `evaluation` in the generation result.

> **Treat this as a trend indicator, not a correctness measure.** It compares *pooled whole-passage* embeddings, so long answers are penalised on `answer_relevance`, and `faithfulness` is easily inflated when the model copies the context verbatim. A fluent but wrong answer that stays on topic can still score well. The grading pipeline in section 4 is the rigorous one.

---

## 4. NLP pipeline: grading answers

### 4.1 Why not just cosine similarity?
The first grader embedded the student's answer and the reference with MiniLM and used `score = round(cosine × 10)`. Embedding similarity measures **relatedness, not correctness**:

- "Supervised learning uses **labeled** data" and "…uses **unlabeled** data" are nearly the same vector, yet contradict each other. Negation, antonyms, swapped roles and wrong numbers all slip through.
- A correct one-sentence answer scores low against a five-sentence reference.
- The reference itself is machine-generated and can be incomplete or wrong, which caps accuracy.
- Unrelated text still scores 0.2–0.4, and the 0.75 pass mark was never calibrated.

### 4.2 Grounded NLI grading
`AnswerGrader` (in `backend.py`) grades against the **source passage** the question came from, using an NLI cross-encoder to judge meaning and embeddings to retrieve evidence and award partial credit. **No keyword overlap, ROUGE/BLEU, n-grams or TF-IDF is used anywhere** — a correct paraphrase scores the same as a verbatim copy.

**Step 1 — Key points (computed once per question, cached).** The reference answer is split into sentences. For each, the three most similar source sentences (plus their concatenation, in document order) are offered as premises. A reference sentence is kept as a graded **key point** only if the source entails it at ≥ `KEYPOINT_GROUNDING_MIN = 0.40`. Unsupported — possibly hallucinated — reference sentences are dropped and reported to the admin as `ungrounded_reference_points`. If *nothing* survives, key points are rebuilt from the two source sentences closest to the question. Near-duplicate key points (cosine ≥ 0.92) are merged.

**Step 2 — Coverage.** For every key point, the student's sentences, and the whole answer as one premise, are tested against it with NLI:

```
closeness_k = clip((max cosine(student text, key point k) − 0.50) / 0.40, 0, 1)

coverage_k  = 0                                   if the student contradicts it (P(contradiction) ≥ 0.60)
            = max( P(entailment),  0.5 × closeness_k )   otherwise

coverage    = mean over key points of coverage_k
```

Full entailment earns full credit; a close paraphrase that doesn't fully entail it earns up to half credit; a contradiction earns nothing and can never be rescued by similarity.

**Step 3 — Support.** Each *on-topic* student sentence (related to the question or the source at cosine ≥ `CLAIM_RELEVANCE_MIN = 0.30`) is verified against its top-3 source sentences (+ their concatenation). Retrieval is by embedding; verification is by NLI. `support` is the mean best entailment. Off-topic filler is ignored — it neither earns nor loses marks. Correct statements the reference happened to omit are therefore *not* treated as errors.

**Step 4 — Combine.**

```
contradiction = strongest contradiction found (key points or claims)
penalty       = 1 − 0.75 × contradiction     if contradiction ≥ 0.60, else 1

quality = coverage × (0.70 + 0.30 × support) × penalty          ∈ [0, 1]
score   = round(10 × quality)          correct  =  quality ≥ 0.60
```

Coverage gates everything — true-but-irrelevant statements can't earn marks on their own. The **entailment override** (`ENTAILMENT_OVERRIDE = 0.50`) discards contradiction signals about any statement that some source sentence clearly entails, which protects correct answers from NLI flagging a near-identical sentence about a different subject (e.g. *un*supervised vs supervised).

### 4.3 Worked example *(illustrative, not measured output)*

Source: *"Supervised learning uses labeled training data. Each training example contains input data and a known target output."* — Question: *"What is supervised learning?"*

| Student answer | What happens | Result |
|---|---|---|
| "It trains on examples where every input has a known correct output, i.e. labeled data." | Both key points entailed, claim supported | high score, `correct` |
| "Supervised learning is a type of machine learning." | On topic but entails neither key point; only partial similarity credit | low score, both points listed in `missing_points` |
| "Supervised learning uses unlabeled data." | Key point contradicted → coverage 0, penalty applied | ≈ 0, claim in `contradicted_claims` |
| "It might rain tomorrow." | Off-topic filler, no relevant claim | 0 |
| (repeats a sentence the *reference* hallucinated) | Sentence isn't in the source → not a key point, not supported | little or no credit — grounding catches it |

### 4.4 What the grader returns

| Field | Meaning |
|---|---|
| `similarity` | Final grounded grade, 0–1 (kept under the old name so existing pages still work; `score == round(similarity × 10)`) |
| `score`, `correct`, `threshold` | 0–10 mark, pass/fail, and the pass mark (0.60) |
| `coverage`, `support`, `contradiction` | The three components above |
| `embedding_similarity` | Plain student↔reference cosine, for comparison with the old method |
| `key_points[]` | Each graded point with its `coverage`, `entailment`, `contradicted` flag |
| `missing_points[]` | Key points with coverage < 0.5 — "what the student missed" |
| `contradicted_claims[]` / `unsupported_claims[]` | Student sentences the source contradicts / doesn't back |
| `ungrounded_reference_points[]` | Reference sentences dropped because the source didn't support them |
| `grounded` | `false` for older material saved without its source passage (graded against the reference only) |
| `note` | Explanation for degenerate cases (empty answer, restates the question, …) |

### 4.5 Safeguards and efficiency
- Empty answers → 0. An answer that just restates the question (cosine ≥ 0.92 to the question and no longer) → 0.
- At most 8 student sentences and 2,500 characters are considered; ≤ 6 key points and 40 source sentences per question.
- NLI pairs below a relevance floor aren't scored; identical pairs are scored once; pairs are batched (16).
- Key-point profiles are cached in memory per (question, reference, source) so later students pay only for their own answer.
- All grading paths use it: exam submission, Quiz Mode and PDF evaluation.

### 4.6 Calibrating it
The thresholds and weights are reasoned defaults, **not measured**. To tune them: hand-grade 20–50 real answers across clearly-right, partial, vague, wrong and contradicting cases; compare against `score`; adjust `GRADE_CORRECT_THRESHOLD`, `SOFT_CREDIT_WEIGHT`, `SUPPORT_WEIGHT`, `CONTRADICTION_THRESHOLD`. A useful behavioural test set: paraphrasing should keep the score, negating a claim should drop it, deleting a key claim should lower it.

---

## 5. Design principles

1. **Meaning over wording.** Only NLI and embeddings decide anything; no lexical matching.
2. **The source is the ground truth.** Generated reference answers can be wrong, so they're checked against — and graded alongside — the original passage.
3. **Retrieve, then verify.** Cheap dense retrieval narrows the evidence; the heavier cross-encoder only runs on a few relevant pairs.
4. **Gate early.** Reject bad questions before spending generation or NLI compute on them.
5. **Explainable by default.** Every grade comes with the key points covered/missed and the contradicted claims.
6. **Fail loudly, degrade gracefully.** Missing models and missing pages raise clear errors; older material without a stored passage still grades.
7. **Local and inspectable.** No hosted LLM, no hidden prompts — all constants sit at the top of `backend.py`.

---

## 6. Known limitations

**Generation**
- FLAN-T5-base is a small model. Questions can be generic and answers often lean extractive. The `difficulty` setting is only a hint in the prompt; nothing verifies that harder questions are harder.
- It can occasionally echo prompt instructions as an "answer". The current answer filter checks only length and question-echo, not instruction leakage.
- The 512-token input limit truncates long contexts from the end, so late chunks may be silently dropped.
- Answer retrieval takes the top-6 chunks with no relevance floor, so a weakly related chunk can reach the generator.
- Question sampling is random: regenerating gives different questions.
- In the generation result, `pages_used` / `retrieval` describe the topic-level retrieval, not every page actually used as a seed.
- PDFs: no OCR; tables, multi-column layouts and figures may extract poorly. English only.

**Grading**
- NLI models make mistakes, including false contradictions and misses on numbers or formulas. Treat grades as decision support for the admin, not final marks.
- Answers that add true statements *outside* the source slightly lower `support`.
- A very terse answer (a single word or phrase) may entail less than its meaning deserves, because the reference key points are full sentences; partial credit softens, but doesn't remove, this.
- Thresholds are uncalibrated (see 4.6).
- Grading runs synchronously inside the submit request — expect roughly a second or two per answer on CPU.

**System**
- JSON files for storage and in-memory login throttling: one server process only.
- Student emails are not verified (no email is sent): whoever registers an approved address first owns that account; the admin can reset it.
- Startup downloads and loads three models, so first start is slow and `--reload` reloads them on every code change.

---

## 7. Getting started

### Requirements
Python 3.10+, and roughly 1–2 GB of disk for model downloads plus a few GB of RAM. A GPU is optional.

```bash
pip install fastapi "uvicorn[standard]" python-multipart itsdangerous python-dotenv reportlab \
            pypdf numpy torch transformers sentence-transformers sentencepiece \
            langchain-community langchain-core
```

`sentencepiece` is needed by the default NLI model; without it the grader falls back to the MiniLM NLI model automatically. `itsdangerous` is required by the session middleware.

### Folder layout
Keep every file in **one folder** — `main.py` serves the HTML pages from its own directory:

```
your-project/
├── main.py            # web app, auth, exams, storage
├── backend.py         # all NLP
├── index.html  materials.html  students.html  review.html  evaluate.html   # admin pages
├── student.html  quiz.html                                                  # student pages
├── login.html  student-login.html
├── .env               # your secrets (never commit)
├── .env.example       # template
├── .gitignore
└── uploads/           # created at runtime (see §11)
```

### Configure
Create `.env` next to `main.py`:

```
ADMIN_USERNAME=your_username
ADMIN_PASSWORD=a_real_password
SESSION_SECRET_KEY=a_long_random_string
```

Without these the app falls back to `admin` / `changeme123` and a random per-restart session key, and logs a warning.

### Run
```bash
uvicorn main:app
```
Start it **from the project folder** (the `uploads/` directory is created relative to where you run it). Add `--reload` only while editing code. The first run downloads the models (roughly 1 GB for FLAN-T5 plus the embedder and NLI models). Then open:

- Admin: `http://127.0.0.1:8000/login`
- Student: `http://127.0.0.1:8000/student-login`

---

## 8. Using the app

**Admin**
1. **Dashboard** — upload a PDF, choose an optional topic, difficulty and number of questions. Generation runs as a background job with progress stages (uploaded → cleaning → chunking → embedding → generating → done).
2. Tick the questions you want and click **Create exam** — on the dashboard right after generating, or on any older material (Dashboard recent list or *My materials*).
3. **Students** — paste approved email addresses (one per line, comma-separated, or `Name <email>`). Share the `/student-login#register` link; nothing is emailed automatically. You can reset a student's password or remove them (access ends immediately).
4. **Review Submissions** — pick an exam to see every submission, then open one for the per-question breakdown.
5. **Answer Evaluation** — grade a student's answers uploaded as a PDF against any material. Answers are split on markers like `Q1:` / `Answer 1:`; with no markers, each PDF page is one answer.
6. Questions can be exported to PDF.

**Student**
1. Open the signup link, choose **Create password** with the approved email, then log in normally afterwards.
2. **Student mode** — choose an open exam, answer each question, submit. Students see a confirmation, not their grade; grading review is on the admin side.
3. **Quiz Mode** — practise any generated set with immediate scoring and weak-area feedback.

---

## 9. API reference

Pages return HTML and redirect to the matching login page when you're not signed in.

### Pages
| Route | Role |
|---|---|
| `/login`, `/student-login` | public |
| `/`, `/my-materials`, `/students`, `/review`, `/evaluate` | admin |
| `/student`, `/quiz` | student |
| `/manage-exams` | redirects to `/` (legacy) |

### Auth & accounts
| Method & route | Access | Purpose |
|---|---|---|
| `POST /login` · `POST /logout` | admin | `{username, password}` |
| `POST /student-register` | public* | First-time password for an *approved* email `{email, name, password}` |
| `POST /student-login` · `POST /student-logout` | public / student | `{email, password}` |
| `GET /me` | any | Current identity |
| `GET /admin/students` | admin | List approved students (never returns hashes) |
| `POST /admin/students` | admin | `{emails: "<pasted text>"}` → `{added, already_listed, invalid}` |
| `DELETE /admin/students/{email}` | admin | Revoke access |
| `POST /admin/students/{email}/reset` | admin | Clear password, back to "invited" |

\*Failed attempts are throttled (5 per 5 minutes → `429`).

### Generation & materials
| Method & route | Access | Purpose |
|---|---|---|
| `POST /upload` | admin | Multipart `file`, `topic`, `difficulty`, `num_questions` → starts a background job |
| `GET /jobs/{id}/status` · `GET /jobs/{id}/result` | admin | Poll progress / fetch the result |
| `POST /generate` | admin | Same inputs, synchronous |
| `GET /materials` · `/materials/{id}` · `/materials/{id}/file` | any login | Saved question sets and the source PDF |
| `POST /export/pdf` | admin | `{material_id}` → PDF of the questions |

### Exams & grading
| Method & route | Access | Purpose |
|---|---|---|
| `POST /admin/exams` | admin | `{material_id, title?, question_indices?}` — snapshots questions, answers **and source passages** |
| `GET /admin/exams` · `GET /admin/exams/{id}` | admin | List / detail |
| `POST /admin/exams/{id}/status` | admin | `{status: "open" \| "closed"}` |
| `GET /admin/exams/{id}/submissions` · `GET /admin/submissions/{id}` | admin | Review |
| `GET /exams` · `GET /exams/{id}` | student | Open exams; **question text only** — never answers or source passages |
| `POST /exams/{id}/submit` | student | `{answers: [{question_index, answer}]}`; identity comes from the session |
| `POST /evaluate-answer` | student | Quiz Mode self-check `{material_id, question_index, student_answer}` |
| `POST /evaluate-submission` | admin | Multipart PDF `file` + `material_id` |

---

## 10. Configuration reference

All in `backend.py` unless noted.

**Chunking & retrieval**

| Constant | Default | Meaning |
|---|---|---|
| `SEMANTIC_SIMILARITY_THRESHOLD` | 0.45 | Adjacent-sentence similarity below which a new chunk starts |
| `MIN_SENTENCES_PER_CHUNK` / `MAX_SENTENCES_PER_CHUNK` | 2 / 6 | Chunk size bounds |
| `TOPIC_TOP_K` | 6 | Minimum seed-pool size / topic retrieval depth |
| `ANSWER_TOP_K` | 6 | Chunks retrieved per question for answering |
| `QUESTION_CONTEXT_WINDOW` | 1 | Neighbouring chunks added around a seed |
| `MIN_SEEDABLE_WORDS` | 12 | Chunks shorter than this can't be seeds |

**Generation**

| Constant | Default | Meaning |
|---|---|---|
| `MAX_INPUT_TOKENS` | 512 | FLAN-T5 input limit |
| `MAX_QUESTION_TOKENS` | 64 | Question length cap |
| `MIN_ANSWER_TOKENS` / `MAX_ANSWER_TOKENS` | 24 / 220 | Answer length bounds |
| `ANSWER_RETRIES_PER_QUESTION` | 2 | Retries on a degenerate answer |

**Question quality gates**

| Constant | Default | Meaning |
|---|---|---|
| `QUESTION_DEDUP_THRESHOLD` | 0.85 | Semantic-duplicate cutoff |
| `MIN_QUESTION_CONTEXT_RELEVANCE` | 0.30 | Answerability gate |

**Grading**

| Constant | Default | Meaning |
|---|---|---|
| `NLI_MODEL_CANDIDATES` | deberta-v3-small → MiniLM2 | Tried in order |
| `GRADE_CORRECT_THRESHOLD` | 0.60 | Pass mark (0–1) |
| `KEYPOINT_GROUNDING_MIN` | 0.40 | Source entailment needed for a reference sentence to count |
| `EVIDENCE_TOP_K` | 3 | Source sentences offered per claim/key point |
| `CONTRADICTION_THRESHOLD` / `CONTRADICTION_PENALTY` | 0.60 / 0.75 | When a contradiction applies and how hard it bites |
| `ENTAILMENT_OVERRIDE` | 0.50 | Entailed statements ignore stray contradiction signals |
| `SOFT_CREDIT_WEIGHT` / `SOFT_SIM_FLOOR` / `SOFT_SIM_CEIL` | 0.50 / 0.50 / 0.90 | Partial credit for close paraphrase |
| `SUPPORT_WEIGHT` | 0.30 | How much unsupported claims can lower a grade |
| `CLAIM_RELEVANCE_MIN` | 0.30 | Below this a sentence is treated as filler |
| `ECHO_SIMILARITY` | 0.92 | Restating-the-question detection |
| `MAX_STUDENT_CLAIMS` / `MAX_STUDENT_CHARS` | 8 / 2500 | Input caps |

**Environment variables** (`main.py`): `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `SESSION_SECRET_KEY`. Login throttling and password hashing (`MAX_FAILED_LOGINS`, `LOGIN_WINDOW_SECONDS`, `PBKDF2_ITERATIONS`, `MIN_PASSWORD_LENGTH`) are also at the top of `main.py`.

---

## 11. Data storage

Everything is plain files in `uploads/`:

| File | Contents |
|---|---|
| `materials.json` | Generated question sets (question, answer, source `context`, citations, difficulty) |
| `exams.json` | Published exams — a snapshot of the chosen questions, reference answers and source passages |
| `submissions.json` | Student submissions with the full grading breakdown for each answer |
| `students.json` | Approved emails and account records (**password hashes — keep private**) |
| `<uuid>_<name>.pdf` | Uploaded source PDFs |

`uploads/` is created relative to the directory you start the server from — start it from the same folder each time or your data will appear to vanish. Back it up and keep it out of version control.

---

## 12. Security notes

- **Admin:** single account from environment variables, timing-safe comparison, throttled after repeated failures.
- **Students:** admin-approved emails; passwords hashed with **PBKDF2-SHA256, 600,000 iterations**, per-user salt; sign-in and registration are throttled; removing or resetting a student ends their session **immediately** (checked on every request).
- Students never receive reference answers or source passages through the exam endpoints. Note that Quiz Mode reads question sets through `/materials`, which includes answers — it is a self-study tool.
- `?next=` redirects accept same-site paths only.
- Not provided: email verification, HTTPS (put a reverse proxy in front for deployment), CSRF tokens beyond `SameSite=Lax` cookies, or multi-process rate limiting.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `File at path …\student.html does not exist` / "page file … is missing" | Put all HTML files in the same folder as `main.py` (the server also logs which pages are missing at startup) |
| `Could not load any NLI cross-encoder model` | `pip install sentencepiece`, and check the machine can reach the Hugging Face hub for the first download |
| Startup error mentioning `itsdangerous` | `pip install itsdangerous` |
| `.env` values ignored | `pip install python-dotenv`; the file must sit next to `main.py` and be named exactly `.env` |
| Materials / students "disappeared" | You started the server from a different folder; run it from the project folder or copy `uploads/` |
| `429 Too many failed attempts` | Wait ~5 minutes, or restart the server (throttle state is in memory) |
| Slow first start / slow reload | Three models load at startup; avoid `--reload` for normal use |
| Weak or repetitive questions | Raise `num_questions` (more seeds), give a `topic`, use a longer/cleaner PDF; tighten `QUESTION_DEDUP_THRESHOLD` or `MIN_QUESTION_CONTEXT_RELEVANCE` |
| Grades feel too harsh / lenient | See 4.6 and adjust `GRADE_CORRECT_THRESHOLD`, `SOFT_CREDIT_WEIGHT`, `CONTRADICTION_THRESHOLD` |

---

## 14. Ideas for future work

- **Calibrate** the grader on admin-overridden grades (logistic/isotonic regression over coverage, support, contradiction, length) and report agreement with human marks (e.g. quadratic weighted kappa).
- **Admin score override** on the review page, doubling as labelled training data.
- **NLI-based faithfulness** for the generation RAG score, replacing pooled-embedding cosine so copied text can't inflate it.
- **Relevance floor + instruction-leak detection** in the answer step; a stronger generator model.
- **Email verification** via a confirmation link, so address ownership is proven.
- **Background grading** with a "grading…" state for large exams.
- **Multiple reference answers** or admin-approved reference answers before publishing.
- **OCR** for scanned PDFs, and multilingual models.
