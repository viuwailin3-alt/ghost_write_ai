import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Ghost Write AI",
    description="Lightweight AI writing API",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    tone: str = Field(default="neutral", max_length=50)


class GenerateResponse(BaseModel):
    prompt: str
    output: str
    model: str


@app.get("/")
def root():
    return {"service": "Ghost Write AI", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # Stub response for local/dev use without an API key configured.
        output = (
            f"[stub] Draft based on your prompt ({request.tone} tone):\n\n"
            f"{request.prompt.strip()}"
        )
        return GenerateResponse(prompt=request.prompt, output=output, model="stub")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
