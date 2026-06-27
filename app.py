#!/usr/bin/env python3
"""
AutoClaw Parallel Swarm UI - Production Web Wrapper
====================================================
Orchestrates 3 specialized AI sub-agents (Architect, Backend, Frontend)
working in parallel via asyncio.gather. Each agent streams live output
to side-by-side Gradio terminal columns with a central execution bus.
"""

import asyncio
import json
import os
import queue
import re
import sys
import textwrap
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import gradio as gr
from openai import AsyncOpenAI

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# ──────────────────────────────────────────────
# Agent Definitions
# ──────────────────────────────────────────────

class AgentRole(Enum):
    ARCHITECT = "architect"
    BACKEND = "backend"
    FRONTEND = "frontend"

AGENT_CONFIGS = {
    AgentRole.ARCHITECT: {
        "name": "🏗️ Architect",
        "color": "#e74c3c",
        "emoji": "🏗️",
        "system_prompt": textwrap.dedent("""\
            You are the **System Architect**. Your job is to design the complete architecture
            for the user's request. Provide:

            1. **High-Level Architecture** — System components, data flow, tech stack
            2. **Component Tree** — How modules relate and communicate
            3. **API / Interface Contracts** — Input/output schemas for every boundary
            4. **Data Model** — Entities, relationships, storage strategy
            5. **Deployment View** — How the system runs in production

            Format your response in clean Markdown with sections.
            Be thorough — your design is the blueprint the other agents build from.
            Always output your full reasoning, then a final "ARCHITECT_PLAN:" section.
        """),
    },
    AgentRole.BACKEND: {
        "name": "⚙️ Backend Dev",
        "color": "#3498db",
        "emoji": "⚙️",
        "system_prompt": textwrap.dedent("""\
            You are the **Backend Developer**. Given the architect's plan, implement:

            1. **Server Setup** — Routes, middleware, error handling
            2. **Database Layer** — ORM models, migrations, queries
            3. **Business Logic** — Services, validators, transformers
            4. **Authentication & Authorization**
            5. **API Endpoints** — Full implementation with request/response handling

            Write production-grade code. Include error handling, logging, and tests.
            Output complete, runnable files in code blocks tagged with the filename.
            Always output your full code, then a final "BACKEND_DONE:" summary.
        """),
    },
    AgentRole.FRONTEND: {
        "name": "🎨 Frontend Dev",
        "color": "#2ecc71",
        "emoji": "🎨",
        "system_prompt": textwrap.dedent("""\
            You are the **Frontend Developer**. Given the architect's plan, build:

            1. **UI Components** — React/Vue/HTML components with full styling
            2. **State Management** — Data flow, stores, hooks
            3. **API Integration** — Fetch hooks, error states, loading states
            4. **Responsive Design** — Mobile-first, accessibility
            5. **Animations & UX** — Transitions, micro-interactions

            Write production-grade code. Include CSS/styling and accessibility.
            Output complete, runnable files in code blocks tagged with the filename.
            Always output your full code, then a final "FRONTEND_DONE:" summary.
        """),
    },
}


# ──────────────────────────────────────────────
# Execution Bus — tracks live state of all agents
# ──────────────────────────────────────────────

@dataclass
class AgentState:
    role: AgentRole
    status: str = "idle"  # idle | running | done | error
    output_lines: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    token_count: int = 0
    error_message: str = ""

    def elapsed(self) -> str:
        if self.start_time is None:
            return "—"
        end = self.end_time or time.time()
        secs = end - self.start_time
        return f"{secs:.1f}s"

    def terminal_text(self) -> str:
        if self.status == "idle":
            return "[Waiting for task...]"
        if self.status == "error":
            return f"❌ ERROR:\n{self.error_message}"
        return "\n".join(self.output_lines) if self.output_lines else "[Running...]"


class ExecutionBus:
    """Central coordinator tracking all agent states and merge conflicts."""

    def __init__(self):
        self.agents: Dict[AgentRole, AgentState] = {
            role: AgentState(role=role) for role in AgentRole
        }
        self.merge_log: List[str] = []
        self.task_id: str = ""
        self.artifacts: Dict[str, str] = {}  # filename -> merged content

    def reset(self, task_id: str):
        self.agents = {role: AgentState(role=role) for role in AgentRole}
        self.merge_log = []
        self.task_id = task_id
        self.artifacts = {}

    def update_agent(self, role: AgentRole, **kwargs):
        agent = self.agents[role]
        for k, v in kwargs.items():
            setattr(agent, k, v)

    def append_output(self, role: AgentRole, text: str):
        agent = self.agents[role]
        for line in text.split("\n"):
            agent.output_lines.append(line)

    def log_merge(self, message: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.merge_log.append(f"[{ts}] {message}")

    def bus_status(self) -> str:
        lines = [f"## 🚦 Execution Bus — Task: `{self.task_id}`", ""]
        lines.append("| Agent | Status | Time | Tokens |")
        lines.append("|-------|--------|------|--------|")
        for role in AgentRole:
            a = self.agents[role]
            cfg = AGENT_CONFIGS[role]
            status_icon = {"idle": "⏳", "running": "🔄", "done": "✅", "error": "❌"}.get(a.status, "❓")
            lines.append(
                f"| {cfg['emoji']} {cfg['name']} | {status_icon} {a.status} | {a.elapsed()} | {a.token_count} |"
            )
        lines.append("")
        if self.merge_log:
            lines.append("### Merge Log")
            for entry in self.merge_log[-20:]:
                lines.append(f"- {entry}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# LLM Streaming Client
# ──────────────────────────────────────────────

class LLMStreamer:
    """Async streaming wrapper around OpenAI-compatible API."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        ) if OPENAI_API_KEY else None

    async def stream_agent(
        self,
        role: AgentRole,
        user_prompt: str,
        context: str = "",
    ) -> AsyncGenerator[Tuple[AgentRole, str], None]:
        """Stream tokens from the LLM for a specific agent role."""
        cfg = AGENT_CONFIGS[role]
        system_msg = cfg["system_prompt"]

        messages = [{"role": "system", "content": system_msg}]
        if context:
            messages.append({"role": "user", "content": f"Context from other agents:\n{context}"})
        messages.append({"role": "user", "content": user_prompt})

        # If no API key, use simulation mode
        if self.client is None:
            for item in self._simulate_stream(role, user_prompt):
                yield item
            return

        try:
            stream = await self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield (role, chunk.choices[0].delta.content)
        except Exception as e:
            yield (role, f"\n\n❌ API Error: {str(e)}")

    def _simulate_stream(
        self, role: AgentRole, user_prompt: str
    ):
        """Simulated streaming for demo/testing without API key."""
        cfg = AGENT_CONFIGS[role]
        sim_texts = {
            AgentRole.ARCHITECT: textwrap.dedent(f"""\
                ## Architecture Analysis

                ### 1. High-Level Architecture
                Based on the task "{user_prompt[:80]}...", I propose a **modular microservices architecture**:

                ```
                ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
                │   Frontend   │────▶│  API Gateway  │────▶│   Services   │
                │  (React/Vite)│     │  (FastAPI)    │     │  (Python)    │
                └─────────────┘     └──────────────┘     └──────────────┘
                ```

                ### 2. Component Tree
                - **UI Layer**: React components with shadcn/ui
                - **State**: Zustand store with persistence
                - **API Layer**: FastAPI with async endpoints
                - **Data Layer**: PostgreSQL with SQLAlchemy
                - **Cache**: Redis for session and rate limiting

                ### 3. API Contracts
                ```yaml
                /api/v1/tasks:
                  GET:    List all tasks (paginated)
                  POST:   Create a new task
                /api/v1/tasks/{{id}}:
                  GET:    Get task details
                  PUT:    Update task
                  DELETE: Delete task
                ```

                ### 4. Data Model
                ```sql
                CREATE TABLE tasks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title VARCHAR(255) NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW()
                );
                ```

                ### 5. Deployment
                - Docker Compose for local dev
                - GitHub Actions CI/CD
                - Deploy to Hugging Face Spaces / Railway

                ARCHITECT_PLAN: Modular microservices with React frontend, FastAPI gateway,
                PostgreSQL database, Redis cache, Docker deployment.
            """),
            AgentRole.BACKEND: textwrap.dedent("""\
                ## Backend Implementation

                ### main.py — FastAPI Server
                ```python
                from fastapi import FastAPI, HTTPException
                from pydantic import BaseModel
                from typing import List, Optional
                import uuid
                from datetime import datetime

                app = FastAPI(title="Task API", version="1.0.0")

                class TaskCreate(BaseModel):
                    title: str
                    description: Optional[str] = None
                    priority: str = "medium"

                class TaskResponse(BaseModel):
                    id: str
                    title: str
                    status: str
                    created_at: datetime

                # In-memory store (replace with DB in production)
                tasks_db: dict = {{}}

                @app.get("/api/v1/tasks", response_model=List[TaskResponse])
                async def list_tasks(page: int = 1, limit: int = 20):
                    offset = (page - 1) * limit
                    items = list(tasks_db.values())[offset:offset + limit]
                    return items

                @app.post("/api/v1/tasks", response_model=TaskResponse, status_code=201)
                async def create_task(task: TaskCreate):
                    task_id = str(uuid.uuid4())
                    now = datetime.utcnow()
                    new_task = {{
                        "id": task_id,
                        "title": task.title,
                        "description": task.description,
                        "priority": task.priority,
                        "status": "pending",
                        "created_at": now,
                    }}
                    tasks_db[task_id] = new_task
                    return TaskResponse(**new_task)

                @app.get("/api/v1/tasks/{{task_id}}", response_model=TaskResponse)
                async def get_task(task_id: str):
                    if task_id not in tasks_db:
                        raise HTTPException(status_code=404, detail="Task not found")
                    return TaskResponse(**tasks_db[task_id])
                ```

                ### Database Layer (SQLAlchemy)
                ```python
                from sqlalchemy import create_engine, Column, String, DateTime
                from sqlalchemy.ext.declarative import declarative_base
                from sqlalchemy.orm import sessionmaker
                import uuid

                Base = declarative_base()

                class TaskModel(Base):
                    __tablename__ = "tasks"
                    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
                    title = Column(String(255), nullable=False)
                    status = Column(String(50), default="pending")
                    created_at = Column(DateTime, default=datetime.utcnow)
                ```

                ### Error Handling
                - Global exception handlers for 400, 404, 500
                - Request validation via Pydantic
                - Structured JSON error responses

                ### Testing
                ```python
                from fastapi.testclient import TestClient
                from main import app
                client = TestClient(app)

                def test_create_task():
                    resp = client.post("/api/v1/tasks", json={{"title": "Test"}})
                    assert resp.status_code == 201
                    assert resp.json()["title"] == "Test"
                ```

                BACKEND_DONE: FastAPI server with CRUD endpoints, SQLAlchemy models,
                Pydantic validation, error handling, and pytest tests.
            """),
            AgentRole.FRONTEND: textwrap.dedent("""\
                ## Frontend Implementation

                ### App.tsx — Main React Component
                ```tsx
                import React, {{ useState, useEffect }} from 'react';
                import {{ Card, CardContent }} from '@/components/ui/card';
                import {{ Button }} from '@/components/ui/button';
                import {{ Input }} from '@/components/ui/input';
                import {{ Badge }} from '@/components/ui/badge';

                interface Task {{
                  id: string;
                  title: string;
                  status: 'pending' | 'in-progress' | 'done';
                  created_at: string;
                }}

                const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

                function App() {{
                  const [tasks, setTasks] = useState<Task[]>([]);
                  const [newTitle, setNewTitle] = useState('');
                  const [loading, setLoading] = useState(false);
                  const [error, setError] = useState<string | null>(null);

                  useEffect(() => {{
                    fetchTasks();
                  }}, []);

                  const fetchTasks = async () => {{
                    try {{
                      setLoading(true);
                      const res = await fetch(`${{API_BASE}}/tasks`);
                      if (!res.ok) throw new Error('Failed to fetch');
                      setTasks(await res.json());
                    }} catch (e: any) {{
                      setError(e.message);
                    }} finally {{
                      setLoading(false);
                    }}
                  }};

                  const createTask = async () => {{
                    if (!newTitle.trim()) return;
                    try {{
                      const res = await fetch(`${{API_BASE}}/tasks`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ title: newTitle }}),
                      }});
                      if (!res.ok) throw new Error('Failed to create');
                      const task = await res.json();
                      setTasks(prev => [task, ...prev]);
                      setNewTitle('');
                    }} catch (e: any) {{
                      setError(e.message);
                    }}
                  }};

                  return (
                    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-8">
                      <header className="mb-8">
                        <h1 className="text-4xl font-bold text-white mb-2">
                          🎛️ Task Manager
                        </h1>
                        <p className="text-slate-400">
                          AutoClaw Parallel Swarm — Built by 3 AI Agents
                        </p>
                      </header>

                      {{/* Create Task Form */}}
                      <Card className="mb-6 bg-slate-800/50 border-slate-700">
                        <CardContent className="pt-6">
                          <div className="flex gap-3">
                            <Input
                              value={{newTitle}}
                              onChange={{(e) => setNewTitle(e.target.value)}}
                              placeholder="What needs to be done?"
                              className="flex-1 bg-slate-700 border-slate-600 text-white"
                              onKeyDown={{(e) => e.key === 'Enter' && createTask()}}
                            />
                            <Button onClick={{createTask}} disabled={{loading}}>
                              Add Task
                            </Button>
                          </div>
                        </CardContent>
                      </Card>

                      {{/* Task List */}}
                      {{error && (
                        <div className="bg-red-900/30 border border-red-700 text-red-300 p-4 rounded-lg mb-4">
                          {{error}}
                        </div>
                      )}}

                      <div className="grid gap-3">
                        {{tasks.map(task => (
                          <Card key={{task.id}}
                            className="bg-slate-800/30 border-slate-700 hover:border-slate-600 transition-colors">
                            <CardContent className="p-4 flex items-center justify-between">
                              <div>
                                <h3 className="text-white font-medium">{{task.title}}</h3>
                                <p className="text-slate-500 text-sm">
                                  {{new Date(task.created_at).toLocaleString()}}
                                </p>
                              </div>
                              <Badge variant={{\\
                                pending: 'secondary',\\
                                'in-progress': 'default',\\
                                done: 'outline'\\
                              }}[task.status] || 'secondary'}>
                                {{task.status}}
                              </Badge>
                            </CardContent>
                          </Card>
                        ))}}
                      </div>

                      {{loading && <p className="text-slate-400 text-center mt-4">Loading...</p>}}
                    </div>
                  );
                }}

                export default App;
                ```

                ### styles.css
                ```css
                @tailwind base;
                @tailwind components;
                @tailwind utilities;

                body {{
                  margin: 0;
                  font-family: 'Inter', system-ui, -apple-system, sans-serif;
                }}
                ```

                ### Responsive Design
                - Mobile-first grid layout
                - Touch-friendly input sizes (min 44px)
                - Dark mode by default with light mode toggle

                FRONTEND_DONE: React + TypeScript + Tailwind app with create/read tasks,
                loading states, error handling, and responsive dark-mode UI.
            """),
        }

        text = sim_texts[role]
        # Stream character by character to simulate real streaming
        for i, char in enumerate(text):
            yield (role, char)
            if i % 5 == 0:
                time.sleep(0.003)  # Simulate network latency

    async def stream_all_agents(
        self,
        user_prompt: str,
    ) -> AsyncGenerator[Dict, None]:
        """Run all 3 agents in parallel and yield state updates."""
        bus = ExecutionBus()
        task_id = f"task-{int(time.time())}"
        bus.reset(task_id)

        # Initialize agents
        for role in AgentRole:
            bus.update_agent(role, status="running", start_time=time.time())

        # Yield initial state
        yield {
            "architect": bus.agents[AgentRole.ARCHITECT].terminal_text(),
            "backend": bus.agents[AgentRole.BACKEND].terminal_text(),
            "frontend": bus.agents[AgentRole.FRONTEND].terminal_text(),
            "bus": bus.bus_status(),
        }

        # Shared queue for streaming tokens from all agents
        token_queue: asyncio.Queue = asyncio.Queue()

        async def run_agent(role: AgentRole):
            """Stream agent output into the shared queue."""
            try:
                async for agent_role, token in self.stream_agent(role, user_prompt):
                    await token_queue.put((agent_role, token))
                    bus.agents[agent_role].token_count += 1
                bus.update_agent(role, status="done", end_time=time.time())
            except Exception as e:
                bus.update_agent(role, status="error", error_message=str(e), end_time=time.time())
            finally:
                await token_queue.put(("__DONE__", role.value))

        # Launch all three agents concurrently
        agent_tasks = [
            asyncio.create_task(run_agent(AgentRole.ARCHITECT)),
            asyncio.create_task(run_agent(AgentRole.BACKEND)),
            asyncio.create_task(run_agent(AgentRole.FRONTEND)),
        ]

        # Process tokens as they arrive, yielding UI updates
        done_count = 0
        last_yield = time.time()
        yield_interval = 0.15  # yield UI updates at most every 150ms

        while done_count < 3:
            try:
                role_val, token = await asyncio.wait_for(
                    token_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                # No new tokens, check if we should still yield a heartbeat
                if time.time() - last_yield >= yield_interval:
                    yield {
                        "architect": bus.agents[AgentRole.ARCHITECT].terminal_text(),
                        "backend": bus.agents[AgentRole.BACKEND].terminal_text(),
                        "frontend": bus.agents[AgentRole.FRONTEND].terminal_text(),
                        "bus": bus.bus_status(),
                    }
                    last_yield = time.time()
                continue

            if role_val == "__DONE__":
                done_count += 1
                bus.log_merge(f"Agent '{token}' completed")
                continue

            # Map string role back to enum
            role = AgentRole(token) if isinstance(token, str) and token not in ("__DONE__",) else role_val
            if isinstance(role_val, AgentRole):
                role = role_val
            elif isinstance(role_val, str) and role_val != "__DONE__":
                continue
            else:
                continue

            bus.append_output(role, token)

            # Throttle UI yields
            if time.time() - last_yield >= yield_interval:
                yield {
                    "architect": bus.agents[AgentRole.ARCHITECT].terminal_text(),
                    "backend": bus.agents[AgentRole.BACKEND].terminal_text(),
                    "frontend": bus.agents[AgentRole.FRONTEND].terminal_text(),
                    "bus": bus.bus_status(),
                }
                last_yield = time.time()

        # Wait for all agent tasks to complete
        await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Extract and merge artifacts
        self._extract_artifacts(bus)
        bus.log_merge("All agents complete — artifacts merged into execution bus")

        # Final yield
        yield {
            "architect": bus.agents[AgentRole.ARCHITECT].terminal_text(),
            "backend": bus.agents[AgentRole.BACKEND].terminal_text(),
            "frontend": bus.agents[AgentRole.FRONTEND].terminal_text(),
            "bus": bus.bus_status(),
        }

    def _extract_artifacts(self, bus: ExecutionBus):
        """Extract code artifacts from agent outputs and detect merge conflicts."""
        artifact_pattern = re.compile(r'###\s+(\S+\.(?:py|tsx?|jsx?|css|yaml|yml|sql|json|html))\s*\n\s*```(?:\w+)?\n(.*?)```', re.DOTALL)

        for role in AgentRole:
            full_output = "\n".join(bus.agents[role].output_lines)
            artifacts = artifact_pattern.findall(full_output)
            for filename, content in artifacts:
                filename = filename.strip()
                content = content.strip()
                if filename in bus.artifacts:
                    bus.log_merge(
                        f"⚠️ MERGE CONFLICT: '{filename}' from {AGENT_CONFIGS[role]['name']} "
                        f"overwrites previous version. Using latest."
                    )
                else:
                    bus.log_merge(f"📦 Artifact '{filename}' registered from {AGENT_CONFIGS[role]['name']}")
                bus.artifacts[filename] = content


# ──────────────────────────────────────────────
# Singleton instances
# ──────────────────────────────────────────────

llm_streamer = LLMStreamer()

# ──────────────────────────────────────────────
# CSS for the Gradio UI
# ──────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ── */
.gradio-container {
    max-width: 100% !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
}
#main-title {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
#main-title h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e74c3c, #3498db, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
#main-title p {
    color: #94a3b8;
    font-size: 0.95rem;
}

/* ── Agent Terminal Panels ── */
.agent-panel {
    border-radius: 12px !important;
    overflow: hidden !important;
}
.agent-panel .panel-header {
    padding: 8px 16px;
    font-weight: 700;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.agent-panel .terminal-output {
    background: #0a0f1a !important;
    color: #00ff88 !important;
    font-size: 0.78rem !important;
    line-height: 1.5 !important;
    padding: 12px !important;
    border-radius: 0 !important;
    min-height: 350px !important;
    max-height: 500px !important;
    overflow-y: auto !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
.agent-panel .terminal-output textarea {
    background: #0a0f1a !important;
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.5 !important;
    border: none !important;
}

/* ── Execution Bus ── */
#bus-panel {
    border: 2px solid #334155 !important;
    border-radius: 12px !important;
    background: #1e293b !important;
    margin-top: 1rem !important;
}
#bus-panel .bus-content {
    background: #0f172a !important;
    color: #f1f5f9 !important;
    font-size: 0.82rem !important;
    padding: 16px !important;
    min-height: 180px !important;
    max-height: 300px !important;
    overflow-y: auto !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

/* ── Buttons ── */
#run-btn {
    background: linear-gradient(135deg, #e74c3c, #f39c12) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    transition: transform 0.15s !important;
    cursor: pointer !important;
}
#run-btn:hover {
    transform: scale(1.03) !important;
}
#run-btn:disabled {
    opacity: 0.5 !important;
    transform: none !important;
}

/* ── Status Pulse ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.pulse {
    animation: pulse 1.5s infinite;
}

/* ── Scrollbar ── */
.terminal-output::-webkit-scrollbar,
.bus-content::-webkit-scrollbar {
    width: 6px;
}
.terminal-output::-webkit-scrollbar-track,
.bus-content::-webkit-scrollbar-track {
    background: #1e293b;
}
.terminal-output::-webkit-scrollbar-thumb,
.bus-content::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 3px;
}
"""


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

def create_ui():
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        title="AutoClaw Parallel Swarm UI",
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div id="main-title">
            <h1>🎛️ AutoClaw Parallel Swarm</h1>
            <p>Three AI agents — Architect, Backend, Frontend — working in parallel. Zero conflicts.</p>
        </div>
        """)

        # ── Input Section ──
        with gr.Row():
            task_input = gr.Textbox(
                label="📋 Task Description",
                placeholder="Describe what you want to build... e.g., 'Build a real-time chat app with user auth and file upload'",
                lines=3,
                scale=6,
                elem_classes=["task-input"],
            )
            with gr.Column(scale=1, min_width=140):
                run_btn = gr.Button(
                    "🚀 Launch Swarm",
                    variant="primary",
                    elem_id="run-btn",
                    size="lg",
                )
                stop_btn = gr.Button(
                    "⏹️ Stop",
                    variant="stop",
                    visible=True,
                    size="sm",
                )
                status_indicator = gr.HTML(
                    value='<div style="color:#94a3b8;text-align:center;padding:8px;">⏳ Ready</div>'
                )

        # ── Agent Terminal Columns ──
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["agent-panel"]):
                gr.HTML(
                    '<div class="panel-header" style="background:#e74c3c;color:white;">🏗️ ARCHITECT</div>'
                )
                architect_output = gr.Textbox(
                    label="",
                    lines=18,
                    interactive=False,
                    elem_classes=["terminal-output"],
                    value="[Waiting for task...]",
                    show_label=False,
                )

            with gr.Column(scale=1, elem_classes=["agent-panel"]):
                gr.HTML(
                    '<div class="panel-header" style="background:#3498db;color:white;">⚙️ BACKEND</div>'
                )
                backend_output = gr.Textbox(
                    label="",
                    lines=18,
                    interactive=False,
                    elem_classes=["terminal-output"],
                    value="[Waiting for task...]",
                    show_label=False,
                )

            with gr.Column(scale=1, elem_classes=["agent-panel"]):
                gr.HTML(
                    '<div class="panel-header" style="background:#2ecc71;color:white;">🎨 FRONTEND</div>'
                )
                frontend_output = gr.Textbox(
                    label="",
                    lines=18,
                    interactive=False,
                    elem_classes=["terminal-output"],
                    value="[Waiting for task...]",
                    show_label=False,
                )

        # ── Execution Bus ──
        gr.HTML(
            '<div class="panel-header" style="background:#1e293b;color:#f1f5f9;border-radius:12px 12px 0 0;margin-top:1rem;border:2px solid #334155;border-bottom:none;">🚦 CENTRAL EXECUTION BUS</div>'
        )
        bus_output = gr.Markdown(
            value="## 🚦 Execution Bus\n\n_No active task. Enter a prompt and launch the swarm._",
            elem_id="bus-panel",
            elem_classes=["bus-content"],
        )

        # ── Event Handler ──
        async def on_run(task_text: str):
            """Run the parallel agent swarm."""
            if not task_text or not task_text.strip():
                yield (
                    "[Error: Please enter a task description]",
                    "[Error: Please enter a task description]",
                    "[Error: Please enter a task description]",
                    "## 🚦 Execution Bus\n\n❌ No task provided.",
                    '<div style="color:#ef4444;text-align:center;padding:8px;">❌ No task</div>',
                )
                return

            yield (
                "▌ Initializing Architect agent...\n▌ Loading system prompt...\n▌ Connecting to LLM...",
                "▌ Initializing Backend agent...\n▌ Loading system prompt...\n▌ Connecting to LLM...",
                "▌ Initializing Frontend agent...\n▌ Loading system prompt...\n▌ Connecting to LLM...",
                "## 🚦 Execution Bus\n\n🔄 **Initializing parallel swarm...**\n\nAll 3 agents launching simultaneously.",
                '<div style="color:#f59e0b;text-align:center;padding:8px;" class="pulse">🔄 Swarm Active</div>',
            )

            try:
                async for state in llm_streamer.stream_all_agents(task_text.strip()):
                    yield (
                        state["architect"],
                        state["backend"],
                        state["frontend"],
                        state["bus"],
                        '<div style="color:#22c55e;text-align:center;padding:8px;" class="pulse">🔄 Running...</div>'
                        if "✅ done" not in state["bus"].lower()
                        else '<div style="color:#22c55e;text-align:center;padding:8px;">✅ Swarm Complete</div>',
                    )
            except Exception as e:
                err_msg = f"❌ Swarm Error: {str(e)}\n\n{traceback.format_exc()}"
                yield (
                    err_msg, err_msg, err_msg,
                    f"## 🚦 Execution Bus\n\n❌ **Error:** {str(e)}",
                    '<div style="color:#ef4444;text-align:center;padding:8px;">❌ Error</div>',
                )

        run_event = run_btn.click(
            fn=on_run,
            inputs=[task_input],
            outputs=[architect_output, backend_output, frontend_output, bus_output, status_indicator],
        )

        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[run_event],
        )

        # ── Example Prompts ──
        gr.Markdown("""
        ### 💡 Try These Prompts
        - **Build a task management app** with React frontend + FastAPI backend + PostgreSQL
        - **Create a real-time chat system** with WebSocket support and user authentication
        - **Design an e-commerce product page** with cart, checkout, and payment integration
        - **Build a CLI tool** for monitoring server health and sending alerts
        """)

    return demo


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────

def main():
    demo = create_ui()
    # Use queue for streaming support
    demo.queue(
        default_concurrency_limit=5,
        max_size=20,
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
