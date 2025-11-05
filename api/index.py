from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse

# Resolve project root (parent of api/)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
MEMORY_DIR = DATA_DIR / "memory"
SCRATCHPAD_PATH = MEMORY_DIR / "scratchpad.md"
MAINPLAN_PATH = MEMORY_DIR / "main-plan.md"
STATE_PATH = DATA_DIR / "state.json"

for p in [DATA_DIR, DOCS_DIR, MEMORY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

if not SCRATCHPAD_PATH.exists():
    SCRATCHPAD_PATH.write_text("", encoding="utf-8")
if not MAINPLAN_PATH.exists():
    MAINPLAN_PATH.write_text("# Main Plan\n\n", encoding="utf-8")
if not STATE_PATH.exists():
    STATE_PATH.write_text(json.dumps({"sessions": {}}, indent=2), encoding="utf-8")

app = FastAPI(title="Agentic Coder API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SaveMemoryRequest(BaseModel):
    scratchpad: Optional[str] = None
    main_plan: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AgentStepRequest(BaseModel):
    session_id: str
    user_input: str
    model: Optional[str] = None
    require_confirmation: bool = True


def load_state() -> Dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"sessions": {}}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


# --- Lightweight TF-IDF search ---
import math
import re
from collections import Counter, defaultdict


class LocalSearchIndex:
    def __init__(self):
        self.documents: List[str] = []
        self.doc_meta: List[Dict[str, Any]] = []
        self.doc_term_freqs: List[Counter[str]] = []
        self.doc_norms: List[float] = []
        self.idf: Dict[str, float] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    def build(self):
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for p in sorted(DOCS_DIR.glob("**/*")):
            if p.is_file():
                try:
                    texts.append(p.read_text(encoding="utf-8"))
                    metas.append({"path": str(p.relative_to(ROOT))})
                except Exception:
                    pass
        try:
            texts.append(SCRATCHPAD_PATH.read_text(encoding="utf-8"))
            metas.append({"path": str(SCRATCHPAD_PATH.relative_to(ROOT))})
        except Exception:
            pass
        try:
            texts.append(MAINPLAN_PATH.read_text(encoding="utf-8"))
            metas.append({"path": str(MAINPLAN_PATH.relative_to(ROOT))})
        except Exception:
            pass

        self.documents = texts
        self.doc_meta = metas
        self.doc_term_freqs = []
        self.doc_norms = []
        df = defaultdict(int)

        for doc in texts:
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for term in tf.keys():
                df[term] += 1

        N = max(len(texts), 1)
        self.idf = {term: math.log((N + 1) / (df_val + 1)) + 1.0 for term, df_val in df.items()}
        self.doc_norms = []
        for tf in self.doc_term_freqs:
            norm_sq = 0.0
            for term, freq in tf.items():
                w = (1 + math.log(freq)) * self.idf.get(term, 0.0)
                norm_sq += w * w
            self.doc_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        vec = {}
        for term, freq in tf.items():
            idf = self.idf.get(term, 0.0)
            if idf == 0:
                continue
            vec[term] = (1 + math.log(freq)) * idf
        return vec

    @staticmethod
    def _cosine(vec: Dict[str, float], tf_doc: Counter[str], idf: Dict[str, float], norm_doc: float) -> float:
        dot = 0.0
        for term, wq in vec.items():
            if term in tf_doc:
                wd = (1 + math.log(tf_doc[term])) * idf.get(term, 0.0)
                dot += wq * wd
        norm_q = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return dot / (norm_q * (norm_doc or 1.0))

    def query(self, q: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.documents:
            self.build()
        if not self.documents:
            return []
        q_tokens = self._tokenize(q)
        q_vec = self._vectorize(q_tokens)
        scores = []
        for i, tf in enumerate(self.doc_term_freqs):
            s = self._cosine(q_vec, tf, self.idf, self.doc_norms[i])
            scores.append((i, s))
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return [{
            "score": float(score),
            "path": self.doc_meta[idx]["path"],
            "snippet": self.documents[idx][:400]
        } for idx, score in ranked]


SEARCH_INDEX = LocalSearchIndex()

# OpenRouter
import httpx
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


async def list_openrouter_models() -> List[Dict[str, Any]]:
    url = f"{OPENROUTER_BASE_URL}/models"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://agentic-2f0b6057.vercel.app"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Agentic Coder"),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        models = data.get("data", data.get("models", []))
        normalized = []
        for m in models:
            pricing = m.get("pricing") or {}
            free = pricing.get("prompt", 0) == 0 and pricing.get("completion", 0) == 0
            normalized.append({
                "id": m.get("id") or m.get("name"),
                "name": m.get("name") or m.get("id"),
                "context_length": m.get("context_length") or m.get("context_length_tokens") or 0,
                "free": bool(free),
                "pricing": pricing,
                "provider": m.get("provider") or {},
            })
        return normalized


async def openrouter_chat(messages: List[Dict[str, str]], model: Optional[str]) -> str:
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://agentic-2f0b6057.vercel.app"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Agentic Coder"),
        "Content-Type": "application/json",
    }
    body = {
        "model": model or "openrouter/auto",
        "messages": messages,
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/models")
async def models():
    try:
        ms = await list_openrouter_models()
    except Exception:
        ms = []
    return {"models": ms}


@app.get("/api/memory")
async def get_memory():
    return {
        "scratchpad": SCRATCHPAD_PATH.read_text(encoding="utf-8"),
        "main_plan": MAINPLAN_PATH.read_text(encoding="utf-8"),
    }


@app.post("/api/memory")
async def save_memory(req: SaveMemoryRequest):
    if req.scratchpad is not None:
        SCRATCHPAD_PATH.write_text(req.scratchpad, encoding="utf-8")
    if req.main_plan is not None:
        MAINPLAN_PATH.write_text(req.main_plan, encoding="utf-8")
    return {"ok": True}


@app.post("/api/docs/upload")
async def upload_doc(file: UploadFile = File(...)):
    content = await file.read()
    safe_name = file.filename.replace("..", "_")
    dest = DOCS_DIR / safe_name
    dest.write_bytes(content)
    return {"ok": True, "path": str(dest.relative_to(ROOT))}


@app.post("/api/search")
async def search(req: SearchRequest):
    SEARCH_INDEX.build()
    return {"results": SEARCH_INDEX.query(req.query, top_k=req.top_k)}


@app.post("/api/agent/step")
async def agent_step(req: AgentStepRequest):
    state = load_state()
    sessions = state.setdefault("sessions", {})
    session = sessions.setdefault(req.session_id, {"messages": [], "approved": False, "model": req.model or None})
    if req.model:
        session["model"] = req.model

    SEARCH_INDEX.build()
    context_snippets = SEARCH_INDEX.query(req.user_input, top_k=5)
    context_text = "\n\n".join([f"Source: {c['path']}\n{c['snippet']}" for c in context_snippets])

    msg_user = (
        "You are an autonomous coding agent with tools and memory. "
        "Leverage the provided CONTEXT and MEMORIES. "
        "If the action may change plans or write files, propose a plan and await explicit APPROVAL if required.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"SCRATCHPAD:\n{SCRATCHPAD_PATH.read_text(encoding='utf-8')}\n\n"
        f"MAIN PLAN:\n{MAINPLAN_PATH.read_text(encoding='utf-8')}\n\n"
        f"USER INPUT:\n{req.user_input}"
    )

    session["messages"].append({"role": "user", "content": msg_user})

    if req.require_confirmation:
        try:
            draft = await openrouter_chat(
                messages=[
                    {"role": "system", "content": "Draft a concise execution plan with numbered steps. Do not execute."},
                    {"role": "user", "content": msg_user},
                ],
                model=session.get("model"),
            )
        except Exception:
            draft = "(Model unavailable. Provide manual approval and proceed.)"
        save_state(state)
        return {"needs_approval": True, "draft_plan": draft, "session_id": req.session_id}

    try:
        reply = await openrouter_chat(
            messages=[{"role": "system", "content": "Execute the plan. Keep responses concise and actionable."}, *session["messages"]],
            model=session.get("model"),
        )
    except Exception as e:
        reply = f"Model call failed: {e}"

    session["messages"].append({"role": "assistant", "content": reply})
    save_state(state)
    return {"needs_approval": False, "reply": reply, "session_id": req.session_id}


@app.get("/")
async def root_html():
    path = ROOT / "public" / "index.html"
    if path.exists():
        return FileResponse(str(path))
    raise HTTPException(status_code=404, detail="UI not found")
