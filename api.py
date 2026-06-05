from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from clasificar import clasificar_comentario


# ── App config ──
app = FastAPI(
    title="Multilingual Comment Classifier",
    description="Classifies movie comments by sentiment (positive/negative/neutral) "
                "and language (en/es/ru/other) using Qwen 2.5 7B running locally via Ollama.",
    version="1.0.0",
)


# ── Pydantic models — validan request y response automáticamente ──
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The comment text to classify")
    prompt_lang: str = Field(default="en", description="Prompt language: 'es' or 'en'")


class ClassifyResponse(BaseModel):
    sentiment: str
    language: str


# ── Endpoints ──
@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "multilingual-comment-classifier"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    """Classify a single comment and return sentiment + detected language."""
    try:
        resultado = clasificar_comentario(req.text, prompt_lang=req.prompt_lang)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
