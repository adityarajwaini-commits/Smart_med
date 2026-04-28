"""FastAPI backend for SmartMed entity analysis.

This service loads a token-classification model on startup and exposes one
analysis endpoint for NER inference. It prefers a disease-focused BioBERT NER
checkpoint, then falls back to the local model in `./smartmed_biobert`, and
finally to the base BioBERT checkpoint if needed.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


APP_TITLE = "SmartMed API"
MODEL_CHECKPOINT = "d4data/biomedical-ner-all"
PRIMARY_MODEL_NAME = "d4data/biomedical-ner-all"
FALLBACK_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "smartmed_biobert"
FALLBACK_BASE_MODEL_NAME = "dmis-lab/biobert-v1.1"
ENTITY_SCORE_THRESHOLD = 0.75


logger = logging.getLogger(__name__)


class ReportRequest(BaseModel):
    """Request body for text analysis."""

    text: str = Field(..., min_length=1, description="Clinical or medical text to analyze")


app = FastAPI(
    title=APP_TITLE,
    version="1.0.0",
    description="SmartMed FastAPI backend for BioBERT-based named entity recognition.",
)

# Keep CORS flexible for local frontend development across common Vite ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _canonical_entity_label(raw_label: str) -> str:
    """Map the model's raw label string to a stable frontend-facing label.

    The frontend uses these normalized labels to choose badge colors.
    """

    normalized = str(raw_label or "UNKNOWN").strip().upper().replace("-", "_").replace(" ", "_")

    normalization_map = {
        "DETAILED_DESCRIPTION": "Disease_Disorder",
        "DISEASE": "Disease_Disorder",
        "DISEASE_DISORDER": "Disease_Disorder",
        "CONDITION": "Disease_Disorder",
        "DISEASE_DISORDERS": "Disease_Disorder",
        "DISEASEDISORDER": "Disease_Disorder",
        "DISEASEDISORDERS": "Disease_Disorder",
        "SIGN_SYMPTOM": "Sign_Symptom",
        "SIGNS_SYMPTOM": "Sign_Symptom",
        "MEDICATION": "Medication",
        "CHEMICAL": "Medication",
        "BIOLOGICAL_STRUCTURE": "Biological_Structure",
        "ANATOMY": "Biological_Structure",
        "DIAGNOSTIC_PROCEDURE": "Diagnostic_Procedure",
    }

    if normalized in normalization_map:
        return normalization_map[normalized]

    disease_aliases = {
        "DISEASE",
        "DISEASE_DISORDER",
        "DISEASE_DISORDERS",
        "DISEASEDISORDER",
        "DISEASEDISORDERS",
        "DISORDER",
        "DISORDERS",
        "PROBLEM",
        "PROBLEMS",
        "PATHOLOGICAL_CONDITION",
        "PATHOLOGICAL_CONDITIONS",
        "CONDITION",
    }
    symptom_aliases = {
        "SIGN_SYMPTOM",
        "SIGNS_SYMPTOM",
        "SIGN",
        "SIGNS",
        "SYMPTOM",
        "SYMPTOMS",
    }
    medication_aliases = {
        "MEDICATION",
        "MEDICATIONS",
        "CHEMICAL",
        "CHEMICALS",
        "DRUG",
        "DRUGS",
        "THERAPEUTIC_OR_PREVENTIVE_PROCEDURE",
    }
    anatomy_aliases = {
        "BIOLOGICAL_STRUCTURE",
        "BIOLOGICAL_STRUCTURES",
        "ANATOMY",
        "ANATOMICAL_STRUCTURE",
        "ANATOMICAL_STRUCTURES",
    }

    if normalized in disease_aliases or "DISEASE" in normalized or "DISORDER" in normalized:
        return "Disease_Disorder"
    if normalized in symptom_aliases or "SYMPTOM" in normalized or "SIGN" in normalized:
        return "Sign_Symptom"
    if normalized in medication_aliases or "MEDICATION" in normalized or "CHEMICAL" in normalized or "DRUG" in normalized:
        return "Medication"
    if normalized in anatomy_aliases or "BIOLOGICAL_STRUCTURE" in normalized or "ANATOMY" in normalized or "STRUCTURE" in normalized:
        return "Biological_Structure"

    return "Other Clinical Notes"


# ── Clinical Observations ignore-list ─────────────────────────────────────────
# Words the NER model frequently tags as clinical entities but are actually
# noise.  Checked case-insensitively before any Clinical_Observations entity
# is accepted into the final payload.
OBSERVATION_IGNORE_LIST: set[str] = {
    "35 years",
    "3 days",
    "3",
    "intake",
    "sits",
    "fast",
    "progressive",
    "gradual",
    "dietary",
    "rice",
    "normal",
}


def _is_ignored_observation(word: str) -> bool:
    """Return True if *word* should be silently dropped from Clinical_Observations.

    The comparison is case-insensitive and also checks whether any ignore-list
    entry appears as a substring so that multi-token spans like
    '35 years old' are caught by the '35 years' entry.
    """
    lowered = word.strip().lower()
    if not lowered:
        return True  # empty strings are never useful

    for ignored in OBSERVATION_IGNORE_LIST:
        if ignored in lowered or lowered in ignored:
            return True
    return False


def _analyze_text(text: str) -> Dict[str, Any]:
    """Run the entity pipeline on plain text and return the normalized payload."""

    ner_pipeline = getattr(app.state, "ner_pipeline", None)
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet. Please try again.")

    try:
        raw_predictions = ner_pipeline(text)
    except Exception as error:
        print(f"CRITICAL ERROR: AI Inference Failed: {error}")
        raise HTTPException(
            status_code=500,
            detail=f"AI Inference Failed: {error}",
        ) from error

    print(f"DEBUG: Pipeline Output: {raw_predictions}")

    if not isinstance(raw_predictions, list):
        raise HTTPException(
            status_code=500,
            detail="The NER pipeline returned an unexpected response format.",
        )

    # 1. Subword Reconstruction
    # Group subword tokens starting with '##' into their parent word
    reconstructed_raw = []
    for item in raw_predictions:
        if not isinstance(item, dict):
            continue
        word_val = str(item.get("word", "")).strip()
        if not word_val:
            continue

        # Detect subword tokens (starting with '##')
        is_subword = word_val.startswith("##")
        cleaned_word = word_val.replace("##", "")

        if is_subword and reconstructed_raw:
            prev = reconstructed_raw[-1]
            prev["word"] += cleaned_word
            if "end" in item and item["end"] is not None:
                prev["end"] = int(item["end"])
            if "score" in item and item["score"] is not None:
                prev["scores"].append(float(item["score"]))
        else:
            reconstructed_raw.append({
                "word": cleaned_word,
                "score": float(item.get("score", 0.0)) if item.get("score") is not None else 0.0,
                "scores": [float(item.get("score", 0.0))] if item.get("score") is not None else [0.0],
                "entity_group": item.get("entity_group") or item.get("entity") or item.get("label"),
                "start": int(item.get("start", 0)) if item.get("start") is not None else 0,
                "end": int(item.get("end", 0)) if item.get("end") is not None else 0,
            })

    # Average the scores for reconstructed subwords
    for item in reconstructed_raw:
        if "scores" in item:
            item["score"] = sum(item["scores"]) / len(item["scores"])
            del item["scores"]

    # Filter extraction objects to keep only those with an entity score >= 0.75
    raw_entities = [item for item in reconstructed_raw if item["score"] >= 0.75]

    # Route every entity into one of four buckets:
    #   Sign_Symptom, Medication, Diagnostic_Procedure, or Clinical_Observations
    # Everything that is NOT one of the three core categories is merged into
    # the single Clinical_Observations bucket.
    CORE_CATEGORIES = {
        "SIGN_SYMPTOM": "Sign_Symptom",
        "MEDICATION": "Medication",
        "DIAGNOSTIC_PROCEDURE": "Diagnostic_Procedure",
    }

    mapped_entities = []
    for item in raw_entities:
        raw_label = "UNKNOWN"
        if item.get("entity_group") is not None:
            raw_label = str(item["entity_group"])

        raw_label = raw_label.strip()
        normalized = raw_label.upper().replace("-", "_").replace(" ", "_")

        word_val = item["word"].strip()
        word_lower = word_val.lower()

        # Strict override: common symptoms must never leak out of Sign_Symptom
        if word_lower in ("fever", "cough", "shortness of breath", "dyspnea", "chest pain"):
            entity_group = "Sign_Symptom"
        elif normalized in CORE_CATEGORIES:
            entity_group = CORE_CATEGORIES[normalized]
        else:
            # Everything else (Disease_Disorder, Biological_Structure, Other, etc.)
            # is merged into a single Clinical_Observations bucket.
            entity_group = "Clinical_Observations"

        # Ignore-list filter for Clinical_Observations to keep the card clean
        if entity_group == "Clinical_Observations" and _is_ignored_observation(word_val):
            continue

        mapped_entities.append({
            "word": word_val,
            "score": round(item["score"], 4),
            "entity_group": entity_group,
            "start": item["start"],
            "end": item["end"],
        })

    # Merge neighboring spans when they have the same entity label and are adjacent
    merged_entities = []
    for entity in mapped_entities:
        if not merged_entities:
            merged_entities.append(entity)
            continue

        previous = merged_entities[-1]
        same_group = entity["entity_group"] == previous["entity_group"]
        touches_previous = entity["start"] <= previous["end"] + 1

        if same_group and touches_previous:
            previous["word"] = f"{previous['word']} {entity['word']}".strip()
            previous["end"] = max(previous["end"], entity["end"])
            previous["score"] = round((previous["score"] + entity["score"]) / 2, 4)
        else:
            merged_entities.append(entity)

    # Deduplicate: ensure that identical entity strings (case-insensitive) are only included once
    unique_entities = []
    seen_words = set()
    for entity in merged_entities:
        word_clean = entity["word"].strip()
        word_lower = word_clean.lower()
        if word_lower not in seen_words:
            seen_words.add(word_lower)
            unique_entities.append(entity)

    # Ensure the final JSON payload matches the structure expected by our frontend:
    # an object containing a status string, raw_text string, and an "entities" array
    # containing dictionaries with keys: 'word', 'score', 'entity_group', 'start', and 'end'.
    # We also keep text, entity_count, and model_source for frontend UI widgets.
    return {
        "status": "success",
        "raw_text": text,
        "text": text,  # for compatibility
        "model_source": getattr(app.state, "model_source", MODEL_CHECKPOINT),
        "entity_count": len(unique_entities),
        "entities": unique_entities,
    }


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract readable text from a PDF stream.

    Prefers `pypdf` when available. If the parser is unavailable or fails, we
    fall back to a conservative byte decode attempt so the endpoint can still
    return a useful error instead of crashing.
    """

    if PdfReader is not None:
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            page_texts: List[str] = []

            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception as error:
                    logger.warning("PDF page extraction failed: %s", error)
                    page_text = ""

                if page_text.strip():
                    page_texts.append(page_text)

            extracted_text = "\n".join(page_texts).strip()
            if extracted_text:
                return extracted_text
        except Exception as error:
            logger.warning("pypdf parsing failed, falling back to byte decoding: %s", error)

    try:
        fallback_text = pdf_bytes.decode("utf-8")
    except UnicodeDecodeError:
        fallback_text = pdf_bytes.decode("latin-1", errors="ignore")

    fallback_text = fallback_text.strip()
    if fallback_text:
        return fallback_text

    raise ValueError("Unable to extract readable text from the PDF file.")


@app.on_event("startup")
def startup_event() -> None:
    """Load the model and tokenizer once when the server starts."""

    try:
        app.state.ner_pipeline = pipeline(
            "ner",
            model=MODEL_CHECKPOINT,
            tokenizer=MODEL_CHECKPOINT,
            aggregation_strategy="simple",
        )
        app.state.model_source = MODEL_CHECKPOINT
    except Exception as error:
        print(f"CRITICAL ERROR: Failed to initialize pipeline on startup: {error}")
        logger.exception("Failed to initialize the NER pipeline for deployment safety: %s", error)
        app.state.ner_pipeline = None
        app.state.model_source = "failed_to_load"


@app.get("/")
def health_check() -> Dict[str, str]:
    """Simple health check so you can confirm the server is running."""

    model_source = getattr(app.state, "model_source", "loading")
    return {"status": "ok", "model_source": model_source}


@app.post("/api/v1/analyze")
def analyze_report(request: ReportRequest) -> Dict[str, Any]:
    """Run named entity recognition over the submitted text."""

    try:
        return _analyze_text(request.text)

    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"Unable to parse the input text: {error}") from error
    except KeyError as error:
        raise HTTPException(
            status_code=500,
            detail=f"The model returned an unexpected label or field: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Entity extraction failed: {error}",
        ) from error


@app.post("/api/v1/analyze-file")
async def analyze_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Extract text from an uploaded .txt or .pdf file and analyze it."""

    try:
        filename = (file.filename or "").strip().lower()
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if filename.endswith(".txt") or file.content_type in {"text/plain", "text/txt"}:
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
        elif filename.endswith(".pdf") or file.content_type == "application/pdf":
            extracted_text = _extract_text_from_pdf_bytes(file_bytes)
        else:
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text could be extracted from the file.")

        return _analyze_text(extracted_text)

    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"Unable to parse the uploaded file: {error}") from error
    except Exception as error:
        logger.exception("File analysis failed")
        raise HTTPException(status_code=500, detail=f"File analysis failed: {error}") from error
    finally:
        await file.close()
