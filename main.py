import shutil
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.concurrency import run_in_threadpool

from backend import generate_questions, DIFFICULTIES

app = FastAPI(
    title="Automatic Question Generator",
    description="PDF based short-answer question generator API",
    version="1.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return {"message": "Automatic Question Generator API", "status": "running"}


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    topic: str = Form(None),
    difficulty: str = Form("medium"),
    num_questions: int = Form(5),
):
    file_path = None
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

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # generate_questions() runs a real NLP pipeline (embeddings + seq2seq
        # generation) which is blocking, CPU/GPU-bound work. Calling it
        # directly inside an `async def` endpoint would block the whole
        # event loop for the duration of generation, stalling every other
        # request (including the health-check endpoint). Offload it to a
        # thread instead.
        result = await run_in_threadpool(
            generate_questions,
            pdf_path=str(file_path),
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions,
        )

        return {"success": True, "data": result}

    except HTTPException:
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
        # Clean up the uploaded file; it was only needed for this request.
        if file_path is not None and file_path.exists():
            try:
                file_path.unlink()
            except OSError:
                pass
