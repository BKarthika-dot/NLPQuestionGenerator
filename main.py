import json
import io
import logging
import re
import shutil
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from starlette.concurrency import run_in_threadpool
from xml.sax.saxutils import escape as escape_xml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("question-lab")

from answer_evaluator import evaluate_answer as score_answer
from backend import DIFFICULTIES, generate_questions, load_pdf

app = FastAPI(
    title="Automatic Question Generator",
    description="PDF based short-answer question generator API",
    version="1.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MATERIALS_FILE = UPLOAD_DIR / "materials.json"
JOBS = {}
JOBS_LOCK = Lock()
JOB_STAGES = ["uploaded", "cleaning", "chunking", "embedding", "generating", "done"]


class AnswerEvaluationRequest(BaseModel):
    material_id: str
    question_index: int
    student_answer: str


class ExportPdfRequest(BaseModel):
    material_id: str


def find_material(material_id):
    return next((item for item in read_materials() if item["id"] == material_id), None)


def compare_answers(expected_answer, student_answer, keywords=None, concepts=None):
    return score_answer(student_answer, expected_answer, keywords=keywords, concepts=concepts)


def update_job(job_id, stage, **values):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update({"stage": stage, **values})
    logger.info("job %s: stage=%s values=%s", job_id, stage, values)


def process_job(job_id, file_path, safe_name, topic, difficulty, num_questions):
    try:
        update_job(job_id, "uploaded", status="processing", progress=10)

        def progress_callback(stage):
            progress = {"cleaning": 25, "chunking": 40, "embedding": 55, "generating": 75}.get(stage, 10)
            update_job(job_id, stage, progress=progress)

        result = generate_questions(
            pdf_path=str(file_path),
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions,
            progress_callback=progress_callback,
        )
        material_id = uuid.uuid4().hex
        materials = read_materials()
        materials.append({
            "id": material_id,
            "filename": safe_name,
            "stored_filename": file_path.name,
            "topic": topic or "General",
            "difficulty": difficulty,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data": result,
        })
        write_materials(materials)
        result["material_id"] = material_id
        update_job(job_id, "done", status="done", progress=100, result=result)
    except Exception as error:
        logger.exception("job %s failed", job_id)
        if file_path.exists():
            file_path.unlink()
        update_job(job_id, "failed", status="failed", progress=0, error=str(error))


def read_materials():
    logger.info("read_materials: start")
    if not MATERIALS_FILE.exists():
        logger.info("read_materials: no materials file")
        return []
    try:
        materials = json.loads(MATERIALS_FILE.read_text(encoding="utf-8"))
        logger.info("read_materials: loaded %d material(s)", len(materials))
        return materials
    except (OSError, json.JSONDecodeError):
        logger.exception("read_materials: failed to read materials file")
        return []


def write_materials(materials):
    logger.info("write_materials: saving %d material(s)", len(materials))
    MATERIALS_FILE.write_text(json.dumps(materials, indent=2), encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def home():
    logger.info("home: serving dashboard")
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


@app.get("/my-materials", response_class=HTMLResponse)
def materials_page():
    logger.info("materials_page: serving materials page")
    return FileResponse(Path(__file__).parent / "materials.html", media_type="text/html")


@app.get("/quiz", response_class=HTMLResponse)
def quiz_page():
    logger.info("quiz_page: serving quiz page")
    return FileResponse(Path(__file__).parent / "quiz.html", media_type="text/html")


@app.get("/evaluate", response_class=HTMLResponse)
def evaluate_page():
    logger.info("evaluate_page: serving evaluation page")
    return FileResponse(Path(__file__).parent / "evaluate.html", media_type="text/html")


@app.get("/materials")
def list_materials():
    logger.info("list_materials: start")
    materials = [
        material for material in read_materials()
        if (UPLOAD_DIR / material["stored_filename"]).exists()
    ]
    logger.info("list_materials: returning %d material(s)", len(materials))
    return {"materials": list(reversed(materials))}


@app.get("/materials/{material_id}")
def get_material(material_id: str):
    logger.info("get_material: id=%s", material_id)
    material = next(
        (item for item in read_materials() if item["id"] == material_id),
        None,
    )
    if not material:
        raise HTTPException(status_code=404, detail="Material not found.")
    return material


@app.get("/materials/{material_id}/file")
def get_material_file(material_id: str):
    logger.info("get_material_file: id=%s", material_id)
    material = next(
        (item for item in read_materials() if item["id"] == material_id),
        None,
    )
    if not material:
        raise HTTPException(status_code=404, detail="Material not found.")
    file_path = UPLOAD_DIR / material["stored_filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found.")
    return FileResponse(file_path, media_type="application/pdf", filename=material["filename"])


@app.post("/evaluate-answer")
def evaluate_answer(payload: AnswerEvaluationRequest):
    material = find_material(payload.material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found.")
    questions = material.get("data", {}).get("questions", [])
    if payload.question_index < 0 or payload.question_index >= len(questions):
        raise HTTPException(status_code=404, detail="Question not found.")
    if not payload.student_answer.strip():
        raise HTTPException(status_code=400, detail="Student answer cannot be empty.")

    question = questions[payload.question_index]
    result = compare_answers(
        question["answer"],
        payload.student_answer,
        keywords=question.get("keywords"),
        concepts=question.get("concepts"),
    )
    return {
        "question_index": payload.question_index,
        "expected_answer": question["answer"],
        **result,
    }


def extract_submission_answers(file_path):
    pages = load_pdf(str(file_path))
    text = "\n".join(document.page_content for document in pages)
    marker_pattern = re.compile(
        r"(?im)(?:^|\n)\s*(?:q(?:uestion)?|answer)\s*(\d{1,3})\s*[:.)-]?\s*"
    )
    matches = list(marker_pattern.finditer(text))
    if not matches:
        matches = list(re.finditer(r"(?m)^\s*(\d{1,3})\s*[.)-]\s*", text))

    if matches:
        answers = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            answer = text[match.end():end].strip()
            if answer:
                answers.append({"question_index": int(match.group(1)) - 1, "answer": answer})
        return answers

    return [
        {"question_index": index, "answer": document.page_content.strip()}
        for index, document in enumerate(pages)
        if document.page_content.strip()
    ]


@app.post("/evaluate-submission")
async def evaluate_submission(
    file: UploadFile = File(...),
    material_id: str = Form(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    material = find_material(material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found.")

    file_path = UPLOAD_DIR / f"evaluation_{uuid.uuid4().hex}_{Path(file.filename).name}"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        extracted_answers = await run_in_threadpool(extract_submission_answers, file_path)
        questions = material.get("data", {}).get("questions", [])
        results = []
        for item in extracted_answers:
            question_index = item["question_index"]
            if question_index < 0 or question_index >= len(questions):
                continue
            question = questions[question_index]
            results.append({
                "question_index": question_index,
                "question": question["question"],
                "expected_answer": question["answer"],
                "student_answer": item["answer"],
                **compare_answers(
                    question["answer"],
                    item["answer"],
                    keywords=question.get("keywords"),
                    concepts=question.get("concepts"),
                ),
            })
        total_score = sum(item["score"] for item in results)
        
        return {
            "material_id": material_id,
            "results": results,
            "total_score": total_score,
            "max_score": len(results) * 10,
            "average_score": round(total_score / len(results), 2) if results else 0,
            "matched_answers": len(results),
            "extraction_note": "Answers are split by markers such as Q1: or Answer 1:. If no markers are found, each PDF page is treated as one answer.",
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    finally:
        if file_path.exists():
            file_path.unlink()


@app.post("/export/pdf")
def export_pdf(payload: ExportPdfRequest):
    material = find_material(payload.material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found.")
    questions = material.get("data", {}).get("questions", [])
    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("Question Lab", styles["Title"]), Paragraph(escape_xml(material.get("filename", "Question paper")), styles["Heading2"]), Spacer(1, 0.2 * inch)]
    for index, question in enumerate(questions, start=1):
        story.append(Paragraph(f"{index}. {escape_xml(question['question'])}", styles["BodyText"]))
        story.append(Spacer(1, 0.18 * inch))
    document.build(story)
    output.seek(0)
    return StreamingResponse(output, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=question-paper.pdf"})


@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    topic: str = Form(None),
    difficulty: str = Form("medium"),
    num_questions: int = Form(5),
):
    logger.info("upload: received filename=%r content_type=%r", file.filename, file.content_type)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    if difficulty not in DIFFICULTIES:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty. Choose from {sorted(DIFFICULTIES)}")
    if num_questions < 1 or num_questions > 50:
        raise HTTPException(status_code=400, detail="num_questions must be between 1 and 50.")

    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "stage": "uploaded",
            "status": "queued",
            "progress": 5,
            "filename": safe_name,
            "error": None,
        }
    background_tasks.add_task(process_job, job_id, file_path, safe_name, topic, difficulty, num_questions)
    logger.info("upload: queued job=%s path=%s", job_id, file_path)
    return {"job_id": job_id, "filename": safe_name}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {key: value for key, value in job.items() if key != "result"}


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job["status"] == "failed":
            raise HTTPException(status_code=500, detail=job.get("error", "Job failed."))
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail="Job is still processing.")
        return {"success": True, "data": job["result"]}


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    topic: str = Form(None),
    difficulty: str = Form("medium"),
    num_questions: int = Form(5),
):
    logger.info(
        "generate: request received filename=%r content_type=%r topic=%r difficulty=%r num_questions=%r",
        file.filename,
        file.content_type,
        topic,
        difficulty,
        num_questions,
    )
    file_path = None
    generation_succeeded = False
    try:
        # Check PDF
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        # Fail fast with a clear message instead of letting the pipeline
        # raise later (nice-to-have, pipeline already validates too).
        if difficulty not in DIFFICULTIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid difficulty. Choose from {sorted(DIFFICULTIES)}",
            )
        if num_questions < 1 or num_questions > 50:
            raise HTTPException(
                status_code=400,
                detail="num_questions must be between 1 and 50.",
            )

        # The uploaded file is saved under a sanitized, UUID-prefixed name
        # rather than the client-supplied filename verbatim. That avoids
        # filename collisions between concurrent uploads and, since the
        # filename is attacker-controlled input, a path-traversal /
        # overwrite risk (e.g. "../../x.pdf").
        safe_name = Path(file.filename).name
        file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
        logger.info("generate: saving upload to %s", file_path)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("generate: upload saved exists=%s size=%d", file_path.exists(), file_path.stat().st_size)

        # generate_questions() runs a real NLP pipeline (embeddings + seq2seq
        # generation) which is blocking, CPU/GPU-bound work. Calling it
        # directly inside an `async def` endpoint would block the whole
        # event loop for the duration of generation, stalling every other
        # request (including the health-check endpoint). Offload it to a
        # thread instead.
        logger.info("generate: starting NLP pipeline")
        result = await run_in_threadpool(
            generate_questions,
            pdf_path=str(file_path),
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions,
        )
        logger.info("generate: NLP pipeline finished generated_questions=%s", result.get("generated_questions"))

        material_id = uuid.uuid4().hex
        materials = read_materials()
        materials.append(
            {
                "id": material_id,
                "filename": safe_name,
                "stored_filename": file_path.name,
                "topic": topic or "General",
                "difficulty": difficulty,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "data": result,
            }
        )
        write_materials(materials)
        generation_succeeded = True
        result["material_id"] = material_id
        logger.info("generate: saved material id=%s", material_id)

        return {"success": True, "data": result}

    except HTTPException:
        logger.exception("generate: HTTP validation failure")
        raise

    except ValueError as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # FIX: this was being caught silently -- HTTPException(detail=str(e))
        # went back to the client, but nothing was ever printed to the
        # server console, so a 500 here gave no way to diagnose it from the
        # uvicorn logs alone. Print the full traceback before converting it.
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if not generation_succeeded and file_path is not None and file_path.exists():
            logger.info("generate: removing failed upload %s", file_path)
            try:
                file_path.unlink()
            except OSError:
                pass
