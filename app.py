import os
import time
import json

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Ghost Write AI",
    description="Lightweight AI writing API + Coding Agent",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_free_models_cache: list[str] = []
_free_models_fetched_at: float = 0
_CACHE_TTL = 300

AGENT_SYSTEM_PROMPT = """You are an expert software engineer and coding agent. When given a task, you:

1. PLAN the task step by step
2. GENERATE complete, working code files
3. EXPLAIN each part clearly

Always respond in this exact JSON format:
{
  "plan": ["step 1", "step 2", "step 3"],
  "files": [
    {
      "filename": "main.py",
      "language": "python",
      "description": "Main application file",
      "content": "# full code here"
    }
  ],
  "explanation": "Clear explanation of what was built and how to run it",
  "next_steps": ["optional next step 1", "optional next step 2"]
}

Rules:
- Always generate COMPLETE, RUNNABLE code (no placeholders like '# TODO')
- Include ALL necessary files (requirements.txt, .env.example, README etc.)
- Code must work immediately after copy-paste
- If the user asks a follow-up question, update only the relevant parts
- For complex projects, break into logical files"""


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


class Message(BaseModel):
    role: str
    content: str


class AgentRequest(BaseModel):
    messages: list[Message]
    task: Optional[str] = None


class CodeFile(BaseModel):
    filename: str
    language: str
    description: str
    content: str


class AgentResponse(BaseModel):
    plan: list[str]
    files: list[CodeFile]
    explanation: str
    next_steps: list[str]
    model: str
    raw: Optional[str] = None


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/agent")
def agent_page():
    return FileResponse("static/agent.html")


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


def _get_client_and_models():
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        primary = os.getenv("OPENROUTER_MODEL", "")
        free_models = _fetch_free_models(openrouter_key)
        if primary and primary not in free_models:
            models = [primary] + free_models
        elif primary:
            models = [primary] + [m for m in free_models if m != primary]
        else:
            models = free_models
        return client, models

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        base_url = os.getenv("OPENAI_BASE_URL")
        kwargs = {"api_key": openai_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        return client, [os.getenv("OPENAI_MODEL", "gpt-4o-mini")]

    return None, []


def _try_completion(client, model: str, messages: list, max_tokens: int = 1024):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or "", completion.model


def _run_with_fallback(client, models: list, messages: list, max_tokens: int = 1024):
    if not models:
        raise HTTPException(status_code=503, detail="No AI models available.")
    last_error = None
    for model in models:
        try:
            return _try_completion(client, model, messages, max_tokens)
        except Exception as exc:
            err = str(exc)
            if any(c in err for c in ["429", "404", "rate", "unavailable", "overloaded", "No endpoints"]):
                last_error = exc
                continue
            raise HTTPException(status_code=502, detail=f"AI provider error: {exc}") from exc
    raise HTTPException(status_code=503, detail=f"All models unavailable. Last error: {last_error}")


def _stream_with_fallback(client, models: list, messages: list, max_tokens: int = 1024):
    if not models:
        raise HTTPException(status_code=503, detail="No AI models available.")
    last_error = None
    for model in models:
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
            )
            return stream, model
        except Exception as exc:
            err = str(exc)
            if any(c in err for c in ["429", "404", "rate", "unavailable", "overloaded", "No endpoints"]):
                last_error = exc
                continue
            raise HTTPException(status_code=502, detail=f"AI provider error: {exc}") from exc
    raise HTTPException(status_code=503, detail=f"All models unavailable. Last error: {last_error}")


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    client, models = _get_client_and_models()
    if not client:
        output = f"[stub] Draft based on your prompt ({request.tone} tone):\n\n{request.prompt.strip()}"
        return GenerateResponse(prompt=request.prompt, output=output, model="stub")

    messages = [
        {"role": "system", "content": f"You are a helpful writing assistant. Tone: {request.tone}."},
        {"role": "user", "content": request.prompt},
    ]
    output, used_model = _run_with_fallback(client, models, messages)
    return GenerateResponse(prompt=request.prompt, output=output, model=used_model)


@app.post("/generate/stream")
def generate_stream(request: GenerateRequest):
    client, models = _get_client_and_models()

    if not client:
        def stub_gen():
            text = f"[stub] Draft based on your prompt ({request.tone} tone):\n\n{request.prompt.strip()}"
            yield f"data: {json.dumps({'type': 'model', 'model': 'stub'})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(stub_gen(), media_type="text/event-stream")

    messages = [
        {"role": "system", "content": f"You are a helpful writing assistant. Tone: {request.tone}."},
        {"role": "user", "content": request.prompt},
    ]

    stream, used_model = _stream_with_fallback(client, models, messages)

    def event_gen():
        yield f"data: {json.dumps({'type': 'model', 'model': used_model})}\n\n"
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield f"data: {json.dumps({'type': 'token', 'text': delta.content})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


class GitHubPushRequest(BaseModel):
    repo: str = Field(..., description="owner/repo format")
    files: list[CodeFile]
    commit_message: str = Field(default="feat: add generated code from Ghost Write AI")
    branch: str = Field(default="main")


class GitHubPushResponse(BaseModel):
    success: bool
    message: str
    commit_url: Optional[str] = None
    pushed_files: list[str] = []


@app.post("/github/push", response_model=GitHubPushResponse)
def github_push(request: GitHubPushRequest):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="GITHUB_TOKEN not configured.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    pushed = []
    last_commit_url = None

    for f in request.files:
        path = f.filename
        url = f"https://api.github.com/repos/{request.repo}/contents/{path}"

        sha = None
        try:
            existing = httpx.get(url, headers=headers, params={"ref": request.branch}, timeout=10)
            if existing.status_code == 200:
                sha = existing.json().get("sha")
        except Exception:
            pass

        import base64
        content_b64 = base64.b64encode(f.content.encode("utf-8")).decode("utf-8")

        payload: dict = {
            "message": request.commit_message,
            "content": content_b64,
            "branch": request.branch,
        }
        if sha:
            payload["sha"] = sha

        try:
            resp = httpx.put(url, headers=headers, json=payload, timeout=15)
            if resp.status_code in (200, 201):
                pushed.append(path)
                last_commit_url = resp.json().get("commit", {}).get("html_url")
            else:
                raise HTTPException(status_code=502, detail=f"GitHub error for {path}: {resp.text}")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to push {path}: {exc}") from exc

    return GitHubPushResponse(
        success=True,
        message=f"Pushed {len(pushed)} file(s) to {request.repo}",
        commit_url=last_commit_url,
        pushed_files=pushed,
    )


@app.post("/agent/chat", response_model=AgentResponse)
def agent_chat(request: AgentRequest):
    client, models = _get_client_and_models()

    if not client:
        return AgentResponse(
            plan=["Stub mode — no API key configured"],
            files=[CodeFile(filename="example.py", language="python", description="Example stub", content='print("Hello from Ghost Write AI Agent!")')],
            explanation="Configure OPENROUTER_API_KEY to enable full agent capabilities.",
            next_steps=["Add your OpenRouter API key"],
            model="stub",
        )

    history = [{"role": m.role, "content": m.content} for m in request.messages]
    system_msg = {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    messages = [system_msg] + history

    raw, used_model = _run_with_fallback(client, models, messages, max_tokens=4000)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        json_str = raw[start:end] if start != -1 else raw
        data = json.loads(json_str)
        files = [CodeFile(**f) for f in data.get("files", [])]
        return AgentResponse(
            plan=data.get("plan", []),
            files=files,
            explanation=data.get("explanation", ""),
            next_steps=data.get("next_steps", []),
            model=used_model,
        )
    except Exception:
        return AgentResponse(
            plan=["Task completed"],
            files=[],
            explanation=raw,
            next_steps=[],
            model=used_model,
            raw=raw,
        )
