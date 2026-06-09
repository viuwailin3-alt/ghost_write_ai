# Ghost Write AI — Project History

## ဘာ Project လဲ
FastAPI နဲ့ ဆောက်ထားတဲ့ AI Writing API။  
OpenRouter (သို့) OpenAI ကို သုံးပြီး prompt ထည့်ရင် AI ရေးပေးတဲ့ service။  
**Goal:** Northflank မှာ deploy လုပ်ဖို့။

---

## Commit မှတ်တမ်း

### `aa625b1` — Add OpenRouter API support
**ဘယ်သူ:** Saw Wai Lin  
**ရက်:** 2026-06-08 (Cursor AI မှာ လုပ်ခဲ့တာ)  
**ဘာလုပ်ခဲ့တယ်:**
- `app.py` ကို ပထမဆုံး ဆောက်ခဲ့တယ် (FastAPI app တစ်ခုလုံး)
- `requirements.txt` ထည့်ခဲ့တယ် (fastapi, uvicorn, gunicorn, pydantic, openai)
- `Procfile` ထည့်ခဲ့တယ် (Northflank/production run command)
- `.python-version` ထည့်ခဲ့တယ်

**API Endpoints:**
- `GET /` → service status
- `GET /health` → health check
- `POST /generate` → AI text generation (prompt + tone လက်ခံတယ်)

**Environment Variables လိုတယ်:**
- `OPENROUTER_API_KEY` → OpenRouter သုံးမယ်ဆိုရင်
- `OPENROUTER_MODEL` → model name (default: `openai/gpt-4o-mini`)
- `OPENAI_API_KEY` → OpenAI တိုက်ရိုက်သုံးမယ်ဆိုရင်

---

### `56f4a7b` / `85e14f3` — Replit Environment Setup
**ဘယ်သူ:** Replit Agent  
**ဘာလုပ်ခဲ့တယ်:**
- Python dependencies တွေ install လုပ်ခဲ့တယ် (`pip install -r requirements.txt`)
- Replit workflow configure လုပ်ခဲ့တယ် (uvicorn port 5000)
- Deployment config set လုပ်ခဲ့တယ် (gunicorn + uvicorn workers, autoscale)

---

## Northflank Deploy လုပ်ဖို့ လိုတာ

1. **Start Command:**
   ```
   gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 2 --timeout 120
   ```

2. **Environment Variables (Northflank ထဲ ထည့်ရမယ်):**
   ```
   OPENROUTER_API_KEY=your_key_here
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```

3. **Postman Test:**
   ```
   POST /generate
   Body: { "prompt": "Write about AI", "tone": "professional" }
   ```

---

## Current Status
- ✅ Postman test — success
- ✅ Replit dev server — running on port 5000
- 🎯 Next: Northflank production deploy
