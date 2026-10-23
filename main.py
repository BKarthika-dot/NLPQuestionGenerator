import hashlib
import hmac
import json
import io
import logging
import os
import re
import secrets
import shutil
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock
from typing import List, Optional
from fastapi import BackgroundTasks, Depends, FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware
from xml.sax.saxutils import escape as escape_xml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("question-lab")

from backend import DIFFICULTIES, generate_questions, load_pdf, pipeline, grade_answer, GRADE_CORRECT_THRESHOLD

app = FastAPI(
    title="Automatic Question Generator",
    description="PDF based short-answer question generator API",
    version="1.0",
)

# ============================================================
# AUTH
#
# Admin: single username/password account, since this app has exactly one
# admin persona (not a multi-teacher system). Set ADMIN_USERNAME /
# ADMIN_PASSWORD as environment variables before using this with real
# users -- the defaults below are only for local testing.
#
# Student: name-only "login". There's no pre-existing student roster to
# check a password against, so this isn't real identity verification --
# it's a session so a student doesn't have to re-type their name on every
# page, and so /exams/{id}/submit and /evaluate-answer are gated behind
# *some* login rather than being wide-open POST endpoints. Anyone who
# knows (or guesses) another student's name can still "log in" as them --
# if that needs to be closed, the next step is a real student roster with
# per-student credentials.
# ============================================================
# Loads a .env file in the project root into os.environ, if one exists and
# python-dotenv is installed. This is optional: if python-dotenv isn't
# installed, .env is simply not read and the code falls back to real
# environment variables / the hardcoded defaults below, exactly as before.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logger.warning(
        "python-dotenv is not installed, so a .env file (if present) will "
        "NOT be loaded -- only real environment variables will be read. "
        "Run: pip install python-dotenv, if you want to use a .env file."
    )

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme123")
SESSION_SECRET_KEY = os.environ.get("SESSION_SECRET_KEY") or secrets.token_hex(32)

if "ADMIN_PASSWORD" not in os.environ:
    logger.warning(
        "ADMIN_PASSWORD is not set -- using the insecure built-in default. "
        "Set ADMIN_USERNAME and ADMIN_PASSWORD env vars before real use."
    )
if "SESSION_SECRET_KEY" not in os.environ:
    logger.warning(
        "SESSION_SECRET_KEY is not set -- using a randomly generated key, "
        "which means every restart logs everyone out. Set SESSION_SECRET_KEY "
        "to a fixed value to keep sessions alive across restarts."
    )

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, same_site="lax")


def current_admin(request: Request) -> bool:
    return bool(request.session.get("is_admin"))


def current_student(request: Request) -> Optional[dict]:
    """The logged-in student's record, or None.

    Re-checked against the approved list on EVERY request, not just at
    login -- so removing a student or resetting their password takes effect
    immediately instead of whenever their cookie happens to expire.
    """
    email = request.session.get("student_email")
    if not email:
        return None
    record = find_student(email)
    if not record or record.get("status") != "registered":
        request.session.pop("student_email", None)
        request.session.pop("student_name", None)
        return None
    return {"email": record["email"], "name": record.get("name") or record["email"]}


def require_admin_api(request: Request) -> None:
    if not current_admin(request):
        raise HTTPException(status_code=401, detail="Admin login required.")


def require_student_api(request: Request) -> dict:
    student = current_student(request)
    if not student:
        raise HTTPException(status_code=401, detail="Student login required.")
    return student


def require_any_login_api(request: Request) -> None:
    if not current_admin(request) and not current_student(request):
        raise HTTPException(status_code=401, detail="Login required.")


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MATERIALS_FILE = UPLOAD_DIR / "materials.json"
EXAMS_FILE = UPLOAD_DIR / "exams.json"
SUBMISSIONS_FILE = UPLOAD_DIR / "submissions.json"
STUDENTS_FILE = UPLOAD_DIR / "students.json"
JOBS = {}
JOBS_LOCK = Lock()
EXAMS_LOCK = Lock()
SUBMISSIONS_LOCK = Lock()
JOB_STAGES = ["uploaded", "cleaning", "chunking", "embedding", "generating", "done"]
QUIZ_CORRECT_THRESHOLD = GRADE_CORRECT_THRESHOLD
EXAM_STATUSES = {"open", "closed"}

# Student accounts: an admin pre-approves email addresses ("invited"); the
# student then sets their own password for that address ("registered").
STUDENTS_LOCK = RLock()
EMAIL_FULL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
PBKDF2_ITERATIONS = 600_000  # OWASP's current recommendation for PBKDF2-HMAC-SHA256
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128  # stops someone submitting a megabyte "password" to burn CPU hashing it
MAX_EMAILS_PER_REQUEST = 500
DUMMY_SALT = secrets.token_bytes(16)

# Failed-login throttling. In-memory, so it resets when the server restarts
# and isn't shared between multiple worker processes -- fine for a single
# uvicorn process, not a substitute for a real rate limiter in production.
FAILED_LOGINS = {}
FAILED_LOGINS_LOCK = Lock()
MAX_FAILED_LOGINS = 5
LOGIN_WINDOW_SECONDS = 300


class AnswerEvaluationRequest(BaseModel):
    material_id: str
    question_index: int
    student_answer: str


class ExportPdfRequest(BaseModel):
    material_id: str


class CreateExamRequest(BaseModel):
    material_id: str
    title: Optional[str] = None
    # None = include every question the material generated. Otherwise the
    # admin has hand-picked which generated questions go into the exam.
    question_indices: Optional[List[int]] = None


class ExamStatusRequest(BaseModel):
    status: str


class SubmitAnswerItem(BaseModel):
    question_index: int
    answer: str = ""


class SubmitExamRequest(BaseModel):
    # Not student_name -- the submitting student's identity comes from
    # their session (set at /student-login), not from a client-supplied
    # field, so a POST to this endpoint can't claim to be a different
    # student than the one who's actually logged in.
    answers: List[SubmitAnswerItem]


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class StudentLoginRequest(BaseModel):
    email: str
    password: str


class StudentRegisterRequest(BaseModel):
    email: str
    name: str = ""
    password: str


class AddStudentsRequest(BaseModel):
    emails: str


def find_material(material_id):
    return next((item for item in read_materials() if item["id"] == material_id), None)


def compare_answers(expected_answer, student_answer, question="", context=""):
    """
    Grade a student answer. `context` is the source passage the question was
    generated from; when present, grading is grounded in it (NLI cross-encoder
    + embeddings, see backend.AnswerGrader), not just in the generated reference.
    """
    return grade_answer(
        question=question,
        reference_answer=expected_answer,
        student_answer=student_answer,
        context=context,
    )


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


def read_exams():
    if not EXAMS_FILE.exists():
        return []
    try:
        return json.loads(EXAMS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("read_exams: failed to read exams file")
        return []


def write_exams(exams):
    with EXAMS_LOCK:
        EXAMS_FILE.write_text(json.dumps(exams, indent=2), encoding="utf-8")


def find_exam(exam_id):
    return next((item for item in read_exams() if item["id"] == exam_id), None)


def read_submissions():
    if not SUBMISSIONS_FILE.exists():
        return []
    try:
        return json.loads(SUBMISSIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("read_submissions: failed to read submissions file")
        return []


def write_submissions(submissions):
    with SUBMISSIONS_LOCK:
        SUBMISSIONS_FILE.write_text(json.dumps(submissions, indent=2), encoding="utf-8")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def read_students():
    if not STUDENTS_FILE.exists():
        return []
    try:
        return json.loads(STUDENTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("read_students: failed to read students file")
        return []


def write_students(students):
    # Write-then-rename so a crash mid-write can't leave a half-written
    # (corrupt) accounts file -- this file holds everyone's password hashes.
    with STUDENTS_LOCK:
        tmp_path = STUDENTS_FILE.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(students, indent=2), encoding="utf-8")
        os.replace(tmp_path, STUDENTS_FILE)


def find_student(email):
    email = normalize_email(email)
    return next((item for item in read_students() if item["email"] == email), None)


def hash_password(password: str, salt: Optional[bytes] = None, iterations: int = PBKDF2_ITERATIONS):
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str, iterations: int) -> bool:
    _, digest = hash_password(password, bytes.fromhex(salt_hex), iterations)
    return hmac.compare_digest(digest, hash_hex)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def check_login_allowed(key: str) -> None:
    now = time.time()
    with FAILED_LOGINS_LOCK:
        recent = [t for t in FAILED_LOGINS.get(key, []) if now - t < LOGIN_WINDOW_SECONDS]
        FAILED_LOGINS[key] = recent
        blocked = len(recent) >= MAX_FAILED_LOGINS
    if blocked:
        raise HTTPException(status_code=429, detail="Too many failed attempts. Please wait a few minutes and try again.")


def record_failed_login(key: str) -> None:
    now = time.time()
    with FAILED_LOGINS_LOCK:
        FAILED_LOGINS.setdefault(key, []).append(now)
        if len(FAILED_LOGINS) > 5000:  # don't let junk keys accumulate forever
            for stale_key in [k for k, v in FAILED_LOGINS.items() if not v or now - v[-1] >= LOGIN_WINDOW_SECONDS]:
                del FAILED_LOGINS[stale_key]


def clear_failed_logins(key: str) -> None:
    with FAILED_LOGINS_LOCK:
        FAILED_LOGINS.pop(key, None)


@app.get("/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    if current_admin(request):
        return RedirectResponse(url="/")
    return FileResponse(Path(__file__).parent / "login.html", media_type="text/html")


@app.post("/login")
def admin_login(request: Request, payload: AdminLoginRequest):
    # compare_digest avoids leaking timing information about how much of
    # the password matched, which a plain == comparison would.
    limiter_key = f"admin:{client_ip(request)}:{payload.username.lower()}"
    check_login_allowed(limiter_key)
    # Encoded to bytes because hmac.compare_digest raises TypeError on
    # non-ASCII str input (which would turn a password with an accent into
    # a 500), and both fields are always compared so a wrong username isn't
    # distinguishable from a wrong password by timing.
    user_ok = hmac.compare_digest(payload.username.encode("utf-8"), ADMIN_USERNAME.encode("utf-8"))
    pass_ok = hmac.compare_digest(payload.password.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8"))
    if not (user_ok and pass_ok):
        record_failed_login(limiter_key)
        logger.info("admin_login: failed attempt for username=%r", payload.username)
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    clear_failed_logins(limiter_key)
    request.session["is_admin"] = True
    request.session["admin_username"] = payload.username
    logger.info("admin_login: success username=%r", payload.username)
    return {"success": True}


@app.post("/logout")
def admin_logout(request: Request):
    request.session.pop("is_admin", None)
    request.session.pop("admin_username", None)
    return {"success": True}


@app.get("/student-login", response_class=HTMLResponse)
def student_login_page(request: Request):
    if current_student(request):
        return RedirectResponse(url="/student")
    return FileResponse(Path(__file__).parent / "student-login.html", media_type="text/html")


def start_student_session(request: Request, record: dict) -> dict:
    request.session["student_email"] = record["email"]
    request.session["student_name"] = record.get("name") or record["email"]
    return {"email": record["email"], "name": record.get("name") or record["email"]}


@app.post("/student-register")
def student_register(request: Request, payload: StudentRegisterRequest):
    """First-time password creation. Only works for an email an admin has
    approved, and only once -- after that the account is "registered" and
    this endpoint refuses it (use /student-login, or ask the admin to reset)."""
    limiter_key = f"register:{client_ip(request)}"
    check_login_allowed(limiter_key)

    email = normalize_email(payload.email)
    name = payload.name.strip()[:80]
    if not MIN_PASSWORD_LENGTH <= len(payload.password) <= MAX_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} characters.",
        )

    with STUDENTS_LOCK:
        students = read_students()
        record = next((item for item in students if item["email"] == email), None)
        if not record or record.get("status") != "invited":
            record_failed_login(limiter_key)
            logger.info("student_register: rejected email=%r", email)
            # Same message whether the email is unknown or already registered,
            # so this endpoint can't be used to probe who is enrolled.
            raise HTTPException(
                status_code=403,
                detail="This email hasn't been approved for access, or it already has an account. "
                       "Try logging in, or ask your instructor.",
            )
        salt_hex, hash_hex = hash_password(payload.password)
        record.update(
            {
                "status": "registered",
                "name": name or record.get("name", ""),
                "salt": salt_hex,
                "password_hash": hash_hex,
                "iterations": PBKDF2_ITERATIONS,
                "registered_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_students(students)

    clear_failed_logins(limiter_key)
    logger.info("student_register: registered email=%r", email)
    return {"success": True, "student": start_student_session(request, record)}


@app.post("/student-login")
def student_login(request: Request, payload: StudentLoginRequest):
    email = normalize_email(payload.email)
    limiter_key = f"student:{client_ip(request)}:{email}"
    check_login_allowed(limiter_key)

    record = find_student(email)
    if record and record.get("status") == "registered" and record.get("password_hash") \
            and len(payload.password) <= MAX_PASSWORD_LENGTH:
        ok = verify_password(payload.password, record["salt"], record["password_hash"],
                             record.get("iterations", PBKDF2_ITERATIONS))
    else:
        # Burn the same CPU a real check would, so response time doesn't
        # reveal whether an email is enrolled.
        hash_password(payload.password[:MAX_PASSWORD_LENGTH], DUMMY_SALT)
        ok = False

    if not ok:
        record_failed_login(limiter_key)
        logger.info("student_login: failed email=%r", email)
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    clear_failed_logins(limiter_key)
    logger.info("student_login: success email=%r", email)
    return {"success": True, "student": start_student_session(request, record)}


@app.post("/student-logout")
def student_logout(request: Request):
    request.session.pop("student_email", None)
    request.session.pop("student_name", None)
    return {"success": True}


@app.get("/me")
def whoami(request: Request):
    student = current_student(request)
    return {
        "is_admin": current_admin(request),
        "student_name": student["name"] if student else None,
        "student_email": student["email"] if student else None,
    }


@app.get("/students", response_class=HTMLResponse)
def students_page(request: Request):
    if not current_admin(request):
        return RedirectResponse(url="/login?next=/students")
    return FileResponse(Path(__file__).parent / "students.html", media_type="text/html")


@app.get("/admin/students", dependencies=[Depends(require_admin_api)])
def list_students():
    students = sorted(read_students(), key=lambda item: item.get("invited_at", ""), reverse=True)
    # Never send salts or hashes to the browser, even to the admin.
    return {
        "students": [
            {
                "email": item["email"],
                "name": item.get("name", ""),
                "status": item["status"],
                "invited_at": item.get("invited_at"),
                "registered_at": item.get("registered_at"),
            }
            for item in students
        ]
    }


@app.post("/admin/students", dependencies=[Depends(require_admin_api)])
def add_students(payload: AddStudentsRequest):
    """Accepts a pasted blob of text (one per line, comma-separated, or
    'Name <email>' style) and approves every address found in it."""
    tokens = [t for t in re.split(r"[\s,;<>\"']+", payload.emails) if "@" in t]
    if not tokens:
        raise HTTPException(status_code=400, detail="No email addresses found. Enter one or more, separated by commas or new lines.")
    if len(tokens) > MAX_EMAILS_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Add at most {MAX_EMAILS_PER_REQUEST} emails at a time.")

    added, already_listed, invalid = [], [], []
    with STUDENTS_LOCK:
        students = read_students()
        known = {item["email"] for item in students}
        for token in tokens:
            email = normalize_email(token)
            if not EMAIL_FULL_RE.match(email):
                invalid.append(token)
            elif email in known:
                if email not in already_listed:
                    already_listed.append(email)
            else:
                students.append(
                    {"email": email, "name": "", "status": "invited",
                     "invited_at": datetime.now().isoformat(timespec="seconds")}
                )
                known.add(email)
                added.append(email)
        if added:
            write_students(students)
    logger.info("add_students: added=%d already=%d invalid=%d", len(added), len(already_listed), len(invalid))
    return {"added": added, "already_listed": already_listed, "invalid": invalid}


@app.delete("/admin/students/{email}", dependencies=[Depends(require_admin_api)])
def remove_student(email: str):
    email = normalize_email(email)
    with STUDENTS_LOCK:
        students = read_students()
        remaining = [item for item in students if item["email"] != email]
        if len(remaining) == len(students):
            raise HTTPException(status_code=404, detail="Student not found.")
        write_students(remaining)
    logger.info("remove_student: email=%r", email)
    return {"success": True}


@app.post("/admin/students/{email}/reset", dependencies=[Depends(require_admin_api)])
def reset_student_password(email: str):
    """Forget the student's password and put the address back to 'invited',
    so they can set a new one. Also immediately ends any session they have."""
    email = normalize_email(email)
    with STUDENTS_LOCK:
        students = read_students()
        record = next((item for item in students if item["email"] == email), None)
        if not record:
            raise HTTPException(status_code=404, detail="Student not found.")
        for field in ("salt", "password_hash", "iterations", "registered_at"):
            record.pop(field, None)
        record["status"] = "invited"
        write_students(students)
    logger.info("reset_student_password: email=%r", email)
    return {"success": True}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if not current_admin(request):
        return RedirectResponse(url="/login?next=/")
    logger.info("home: serving dashboard")
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


@app.get("/my-materials", response_class=HTMLResponse)
def materials_page(request: Request):
    if not current_admin(request):
        return RedirectResponse(url="/login?next=/my-materials")
    logger.info("materials_page: serving materials page")
    return FileResponse(Path(__file__).parent / "materials.html", media_type="text/html")


@app.get("/quiz", response_class=HTMLResponse)
def quiz_page(request: Request):
    if not current_student(request):
        return RedirectResponse(url="/student-login?next=/quiz")
    logger.info("quiz_page: serving quiz page")
    return FileResponse(Path(__file__).parent / "quiz.html", media_type="text/html")


@app.get("/evaluate", response_class=HTMLResponse)
def evaluate_page(request: Request):
    if not current_admin(request):
        return RedirectResponse(url="/login?next=/evaluate")
    logger.info("evaluate_page: serving evaluation page")
    return FileResponse(Path(__file__).parent / "evaluate.html", media_type="text/html")


@app.get("/manage-exams")
def manage_exams_page_redirect():
    # Exam creation now lives inline on the Dashboard and My Materials pages
    # (selecting questions right where they're generated/browsed, instead
    # of a separate page), so this is kept only as a redirect for anyone
    # with the old URL bookmarked.
    logger.info("manage_exams_page_redirect: redirecting legacy /manage-exams to /")
    return RedirectResponse(url="/", status_code=302)


@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request):
    if not current_admin(request):
        return RedirectResponse(url="/login?next=/review")
    logger.info("review_page: serving admin submission review page")
    return FileResponse(Path(__file__).parent / "review.html", media_type="text/html")


@app.get("/student", response_class=HTMLResponse)
def student_page(request: Request):
    if not current_student(request):
        return RedirectResponse(url="/student-login?next=/student")
    logger.info("student_page: serving student mode page")
    return FileResponse(Path(__file__).parent / "student.html", media_type="text/html")


@app.get("/materials", dependencies=[Depends(require_any_login_api)])
def list_materials():
    logger.info("list_materials: start")
    materials = [
        material for material in read_materials()
        if (UPLOAD_DIR / material["stored_filename"]).exists()
    ]
    logger.info("list_materials: returning %d material(s)", len(materials))
    return {"materials": list(reversed(materials))}


@app.get("/materials/{material_id}", dependencies=[Depends(require_any_login_api)])
def get_material(material_id: str):
    logger.info("get_material: id=%s", material_id)
    material = next(
        (item for item in read_materials() if item["id"] == material_id),
        None,
    )
    if not material:
        raise HTTPException(status_code=404, detail="Material not found.")
    return material


@app.get("/materials/{material_id}/file", dependencies=[Depends(require_any_login_api)])
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



# ============================================================
# EXAM MODE (admin: create/manage exams + review submissions;
#            student: take an open exam and submit answers)
#
# NOTE: there is no login/auth system in this app. "Admin" and "student"
# here are separate UI surfaces and endpoint namespaces (/admin/... vs
# /exams, /student), not an access-controlled security boundary -- anyone
# with the URL can currently reach either one. If this needs to be locked
# down for real users, the next step is adding actual authentication in
# front of the /admin/* routes and the Dashboard, My Materials, and
# /review pages (exam creation now lives inline on the first two).
# ============================================================

@app.post("/admin/exams", dependencies=[Depends(require_admin_api)])
def create_exam(payload: CreateExamRequest):
    material = find_material(payload.material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found.")

    questions = material.get("data", {}).get("questions", [])
    if not questions:
        raise HTTPException(status_code=400, detail="This material has no generated questions.")

    if payload.question_indices is not None:
        if not payload.question_indices:
            raise HTTPException(status_code=400, detail="Select at least one question.")
        selected = []
        for index in payload.question_indices:
            if index < 0 or index >= len(questions):
                raise HTTPException(status_code=400, detail=f"Question index {index} is out of range.")
            selected.append(questions[index])
    else:
        selected = questions

    exam_id = uuid.uuid4().hex
    exam = {
        "id": exam_id,
        "material_id": payload.material_id,
        "source_filename": material.get("filename"),
        "title": (payload.title or "").strip() or f"Exam · {material.get('filename', 'Untitled')}",
        # "open" exams are visible to students and accept submissions;
        # "closed" ones stop accepting submissions but submissions already
        # collected stay reviewable by the admin.
        "status": "open",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Snapshotting the questions (including the reference answers used
        # for grading) onto the exam itself, rather than reading live from
        # the material, means a later regeneration/edit of the source
        # material can't silently change an exam students have already
        # started answering.
        "questions": [
            {
                "question": item["question"],
                "answer": item["answer"],
                "difficulty": item.get("difficulty"),
                "type": item.get("type"),
                "context": item.get("context", ""),
                "source": item.get("source", []),
            }
            for item in selected
        ],
    }
    exams = read_exams()
    exams.append(exam)
    write_exams(exams)
    logger.info("create_exam: id=%s questions=%d", exam_id, len(exam["questions"]))
    return exam


@app.get("/admin/exams", dependencies=[Depends(require_admin_api)])
def list_exams_admin():
    exams = read_exams()
    submissions = read_submissions()
    enriched = []
    for exam in reversed(exams):
        submission_count = sum(1 for item in submissions if item["exam_id"] == exam["id"])
        enriched.append(
            {
                "id": exam["id"],
                "title": exam["title"],
                "status": exam["status"],
                "created_at": exam["created_at"],
                "source_filename": exam.get("source_filename"),
                "question_count": len(exam["questions"]),
                "submission_count": submission_count,
            }
        )
    return {"exams": enriched}


@app.get("/admin/exams/{exam_id}", dependencies=[Depends(require_admin_api)])
def get_exam_admin(exam_id: str):
    exam = find_exam(exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found.")
    return exam


@app.post("/admin/exams/{exam_id}/status", dependencies=[Depends(require_admin_api)])
def set_exam_status(exam_id: str, payload: ExamStatusRequest):
    if payload.status not in EXAM_STATUSES:
        raise HTTPException(status_code=400, detail=f"status must be one of {sorted(EXAM_STATUSES)}.")
    exams = read_exams()
    exam = next((item for item in exams if item["id"] == exam_id), None)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found.")
    exam["status"] = payload.status
    write_exams(exams)
    logger.info("set_exam_status: id=%s status=%s", exam_id, payload.status)
    return exam


@app.get("/exams", dependencies=[Depends(require_student_api)])
def list_exams_student():
    """Open exams only, and only the fields a student is allowed to see
    before starting one -- no reference answers, no question text here."""
    exams = [exam for exam in read_exams() if exam["status"] == "open"]
    return {
        "exams": [
            {
                "id": exam["id"],
                "title": exam["title"],
                "created_at": exam["created_at"],
                "question_count": len(exam["questions"]),
            }
            for exam in reversed(exams)
        ]
    }


@app.get("/exams/{exam_id}", dependencies=[Depends(require_student_api)])
def get_exam_student(exam_id: str):
    exam = find_exam(exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found.")
    if exam["status"] != "open":
        raise HTTPException(status_code=403, detail="This exam is closed and no longer available.")
    return {
        "id": exam["id"],
        "title": exam["title"],
        # Only the question text goes to the student -- reference answers
        # and source/citation metadata stay server-side.
        "questions": [{"question": item["question"]} for item in exam["questions"]],
    }


@app.post("/exams/{exam_id}/submit")
def submit_exam(exam_id: str, payload: SubmitExamRequest, student: dict = Depends(require_student_api)):
    exam = find_exam(exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found.")
    if exam["status"] != "open":
        raise HTTPException(status_code=403, detail="This exam is closed and no longer accepting submissions.")
    if not payload.answers:
        raise HTTPException(status_code=400, detail="No answers were submitted.")

    questions = exam["questions"]
    answer_by_index = {item.question_index: item.answer.strip() for item in payload.answers}

    # Automated evaluation happens here, at submission time, using the same
    # embedding-similarity grading already used by Quiz Mode -- so the
    # admin doesn't have to trigger grading separately later.
    results = []
    for index, question in enumerate(questions):
        student_answer = answer_by_index.get(index, "")
        if student_answer:
            evaluation = compare_answers(
                question["answer"], student_answer,
                question=question["question"], context=question.get("context", ""),
            )
        else:
            evaluation = {"similarity": 0.0, "score": 0, "correct": False, "threshold": QUIZ_CORRECT_THRESHOLD}
        results.append(
            {
                "question_index": index,
                "question": question["question"],
                "expected_answer": question["answer"],
                "student_answer": student_answer,
                **evaluation,
            }
        )

    total_score = sum(item["score"] for item in results)
    submission = {
        "id": uuid.uuid4().hex,
        "exam_id": exam_id,
        "exam_title": exam["title"],
        "student_name": student["name"],
        "student_email": student["email"],
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "total_score": total_score,
        "max_score": len(results) * 10,
        "average_score": round(total_score / len(results), 2) if results else 0,
    }
    submissions = read_submissions()
    submissions.append(submission)
    write_submissions(submissions)
    logger.info(
        "submit_exam: exam=%s student=%r score=%s/%s",
        exam_id, submission["student_name"], total_score, submission["max_score"],
    )

    # The student gets confirmation that the submission was received, not
    # the score itself -- this is a formal exam flow, so grading review
    # happens on the admin side via /review, not immediately for the
    # student the way Quiz Mode's self-check does.
    return {
        "submission_id": submission["id"],
        "submitted_at": submission["submitted_at"],
        "question_count": len(results),
    }


@app.get("/admin/exams/{exam_id}/submissions", dependencies=[Depends(require_admin_api)])
def list_submissions(exam_id: str):
    exam = find_exam(exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found.")
    submissions = [item for item in read_submissions() if item["exam_id"] == exam_id]
    submissions.sort(key=lambda item: item["submitted_at"], reverse=True)
    return {
        "exam": {"id": exam["id"], "title": exam["title"], "status": exam["status"]},
        "submissions": [
            {
                "id": item["id"],
                "student_name": item["student_name"],
                "student_email": item.get("student_email", ""),
                "submitted_at": item["submitted_at"],
                "total_score": item["total_score"],
                "max_score": item["max_score"],
                "average_score": item["average_score"],
            }
            for item in submissions
        ],
    }


@app.get("/admin/submissions/{submission_id}", dependencies=[Depends(require_admin_api)])
def get_submission(submission_id: str):
    submission = next((item for item in read_submissions() if item["id"] == submission_id), None)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found.")
    return submission


@app.post("/evaluate-answer", dependencies=[Depends(require_student_api)])
def evaluate_answer(payload: AnswerEvaluationRequest):
    material = find_material(payload.material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found.")
    questions = material.get("data", {}).get("questions", [])
    if payload.question_index < 0 or payload.question_index >= len(questions):
        raise HTTPException(status_code=404, detail="Question not found.")
    if not payload.student_answer.strip():
        raise HTTPException(status_code=400, detail="Student answer cannot be empty.")

    target = questions[payload.question_index]
    result = compare_answers(
        target["answer"], payload.student_answer,
        question=target["question"], context=target.get("context", ""),
    )
    return {
        "question_index": payload.question_index,
        "expected_answer": questions[payload.question_index]["answer"],
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


@app.post("/evaluate-submission", dependencies=[Depends(require_admin_api)])
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
                    question["answer"], item["answer"],
                    question=question["question"], context=question.get("context", ""),
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


@app.post("/export/pdf", dependencies=[Depends(require_admin_api)])
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


@app.post("/upload", dependencies=[Depends(require_admin_api)])
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


@app.get("/jobs/{job_id}/status", dependencies=[Depends(require_admin_api)])
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {key: value for key, value in job.items() if key != "result"}


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_admin_api)])
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


@app.post("/generate", dependencies=[Depends(require_admin_api)])
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