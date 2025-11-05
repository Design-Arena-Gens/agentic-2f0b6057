import asyncio
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Health
r = client.get('/api/health')
print('health', r.status_code, r.json())

# Memory get
r = client.get('/api/memory')
print('memory-get', r.status_code, list(r.json().keys()))

# Memory save
r = client.post('/api/memory', json={'scratchpad':'hello', 'main_plan':'# Plan\n- Step'})
print('memory-post', r.status_code, r.json())

# Search
r = client.post('/api/search', json={'query':'hello', 'top_k': 3})
print('search', r.status_code, r.json())

# Agent draft (no OPENROUTER key, should gracefully handle)
r = client.post('/api/agent/step', json={'session_id':'t1','user_input':'create api','require_confirmation': True})
print('agent-draft', r.status_code, r.json())
