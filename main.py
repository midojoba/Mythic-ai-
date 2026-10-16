import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Mythic AI")

# System Persona for Mythic AI
SYSTEM_PROMPT = (
    "You are Mythic AI, an intelligent, sharp, and visionary AI assistant. "
    "You provide clear, insightful, and well-structured answers with a touch of wit. "
    "Use Markdown formatting (bolding, bullet points, code blocks) when helpful."
)

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    provider: str = "auto"  # "gemini", "claude", or "auto"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    provider = req.provider
    if provider == "auto":
        if gemini_key:
            provider = "gemini"
        elif anthropic_key:
            provider = "claude"
        else:
            return {
                "reply": "⚡ **Mythic AI Demo Mode:** No API key detected yet! Add `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` to your environment variables (Secrets) to unlock my full brain.",
                "provider_used": "demo"
            }

    try:
        # Route 1: Google Gemini API
        if provider == "gemini":
            if not gemini_key:
                raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not set.")
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            contents = []
            for m in req.messages:
                role = "user" if m.role == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.content)]))

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                )
            )
            return {"reply": response.text, "provider_used": "Gemini 2.5 Flash"}

        # Route 2: Anthropic Claude API
        elif provider == "claude":
            if not anthropic_key:
                raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY is not set.")
            import anthropic

            client = anthropic.Anthropic(api_key=anthropic_key)
            formatted_messages = [{"role": m.role, "content": m.content} for m in req.messages]

            response = client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=formatted_messages
            )
            return {"reply": response.content[0].text, "provider_used": "Claude Sonnet"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
