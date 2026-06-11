import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(
    title="Ghost Write AI",
    description="Lightweight AI writing API",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_free_models_cache: list[str] = []
_free_models_fetched_at: float = 0
_CACHE_TTL = 300


def _fetch_free_models(api_key: str) -> list[str]:
    global _free_models_cache, _free_models_fetched_at
    now = time.time()
    if _free_models_cache and (now - _free_models_fetched_at) < _CACHE_TTL:
        return _free_models_cache
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        data = resp.json().get("data", [])
        free = [m["id"] for m in data if str(m.get("id", "")).endswith(":free")]
        if free:
            _free_models_cache = free
            _free_models_fetched_at = now
    except Exception:
        pass
    return _free_models_cache


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    tone: str = Field(default="neutral", max_length=50)


class GenerateResponse(BaseModel):
    prompt: str
    output: str
    model: str


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/status")
def status():
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openrouter_key:
        model = os.getenv("OPENROUTER_MODEL", "auto")
        free_models = _fetch_free_models(openrouter_key)
        return {"provider": "openrouter", "primary_model": model, "available_free_models": free_models[:10]}
    if openai_key:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return {"provider": "openai", "model": model}
    return {"provider": "stub", "model": "stub"}


def _get_ai_client():
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        ), os.getenv("OPENROUTER_MODEL", ""), openrouter_key

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        base_url = os.getenv("OPENAI_BASE_URL")
        client_kwargs = {"api_key": openai_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        return OpenAI(**client_kwargs), os.getenv("OPENAI_MODEL", "gpt-4o-mini"), None

    return None, None, None


def _try_completion(client, model: str, messages: list) -> tuple[str, str]:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
    )
    return completion.choices[0].message.content or "", completion.model


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    client, model, openrouter_key = _get_ai_client()

    if not client:
        output = (
            f"[stub] Draft based on your prompt ({request.tone} tone):\n\n"
            f"{request.prompt.strip()}"
        )
        return GenerateResponse(prompt=request.prompt, output=output, model="stub")

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful writing assistant. Tone: {request.tone}.",
        },
        {"role": "user", "content": request.prompt},
    ]

    if openrouter_key:
        free_models = _fetch_free_models(openrouter_key)
        if model and model not in free_models:
            models_to_try = [model] + free_models
        elif model:
            models_to_try = [model] + [m for m in free_models if m != model]
        else:
            models_to_try = free_models
    else:
        models_to_try = [model]

    if not models_to_try:
        raise HTTPException(status_code=503, detail="No AI models available. Please try again later.")

    last_error = None
    for candidate in models_to_try:
        try:
            output, used_model = _try_completion(client, candidate, messages)
            return GenerateResponse(prompt=request.prompt, output=output, model=used_model)
        except Exception as exc:
            err_str = str(exc)
            if any(code in err_str for code in ["429", "404", "rate", "unavailable", "overloaded", "No endpoints"]):
                last_error = exc
                continue
            raise HTTPException(status_code=502, detail=f"AI provider error: {exc}") from exc

    raise HTTPException(
        status_code=503,
        detail=f"All {len(models_to_try)} models are currently unavailable. Please try again in a few minutes.",
    )
