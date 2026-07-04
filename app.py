from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re
import os
import traceback

# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title="SummarAI",
    description="Text Summarization using BART",
    version="1.0",
)

# ── Device ───────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[SummarAI] Using device: {device}")

# ── Load model once at startup ───────────────────────────
MODEL_PATH = "Yag06/text_summarization"  # Loaded from HF Hub at runtime
MODEL_READY = False
MODEL_ERROR = ""

# Read optional HF token (set as a Secret in HF Spaces settings)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN:
    print("[SummarAI] HF_TOKEN found — will use for authenticated model access.")
else:
    print("[SummarAI] No HF_TOKEN — attempting public model access.")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, token=HF_TOKEN)
    model.to(device)
    model.eval()          # inference mode — disables dropout
    MODEL_READY = True
    print("[SummarAI] Model loaded successfully.")
except Exception as e:
    MODEL_ERROR = traceback.format_exc()
    print(f"[SummarAI] ERROR — could not load model: {e}")
    print(MODEL_ERROR)

# ── Templates ─────────────────────────────────────────────
templates = Jinja2Templates(directory=".")

# ── Request schema ────────────────────────────────────────
class DialogueInput(BaseModel):
    dialogue: str

# ── Helpers ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove newlines, extra spaces, and HTML tags before tokenising."""
    text = re.sub(r"\r\n|\r|\n", " ", text)   # line breaks → space
    text = re.sub(r"<.*?>", " ", text)          # strip HTML tags
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    return text.strip().lower()


def run_summarize(dialogue: str) -> str:
    """Tokenise, generate, and decode the summary."""
    cleaned = clean_text(dialogue)

    inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():                        # saves memory, speeds up inference
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# ── Routes ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the frontend UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    """
    Accept text and return an AI-generated summary.

    Request body : { "dialogue": "<text>" }
    Response     : { "summary": "<summary>" }
    """
    # Check model is available
    if not MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check the server logs.",
        )

    text = dialogue_input.dialogue.strip()

    # Validate input
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty.",
        )

    if len(text) > 20_000:
        raise HTTPException(
            status_code=400,
            detail="Input text is too long (max 20,000 characters).",
        )

    # Run inference
    try:
        summary = run_summarize(text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}",
        )

    return {"summary": summary}


@app.get("/health")
async def health():
    """Quick health-check used by the frontend status indicator."""
    return {
        "status": "ok",
        "model_ready": MODEL_READY,
        "device": str(device),
    }


@app.get("/logs")
async def logs():
    """Diagnostic endpoint — shows model loading error if any."""
    return {
        "model_ready": MODEL_READY,
        "model_path": MODEL_PATH,
        "device": str(device),
        "hf_token_set": HF_TOKEN is not None,
        "error": MODEL_ERROR if MODEL_ERROR else "None",
    }
