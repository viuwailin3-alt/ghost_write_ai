import os

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


def _get_ai_client():
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        ), os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        base_url = os.getenv("OPENAI_BASE_URL")
        client_kwargs = {"api_key": openai_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        return OpenAI(**client_kwargs), os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    return None, None


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    client, model = _get_ai_client()

    if not client:
        output = (
            f"[stub] Draft based on your prompt ({request.tone} tone):\n\n"
            f"{request.prompt.strip()}"
        )
        return GenerateResponse(prompt=request.prompt, output=output, model="stub")

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful writing assistant. Tone: {request.tone}.",
                },
                {"role": "user", "content": request.prompt},
            ],
            max_tokens=1024,
        )
        output = completion.choices[0].message.content or ""
        return GenerateResponse(
            prompt=request.prompt,
            output=output,
            model=completion.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}") from exc
