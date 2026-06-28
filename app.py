#!/usr/bin/env python3
"""
AutoClaw Unlimited Multi-Agent Swarm Orchestrator
==================================================
Dynamic agent registry — add unlimited agents, configure unlimited critic loops,
and use specialized models including JavaScript (Peter JS).

Swarm flow:  Generator → Critics → (Rejected → Refactorers) ↩ max N loops → QA → [VERIFIED]
"""

import asyncio
import copy
import hashlib
import json
import os
import re
import sys
import textwrap
import time
import requests
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import gradio as gr
from openai import AsyncOpenAI


# ═══════════════════════════════════════════════════════════════
# AGENT TEMPLATES — pre-built agent definitions
# ═══════════════════════════════════════════════════════════════

class RoleType(Enum):
    GENERATOR   = "generator"    # produces code
    CRITIC      = "critic"       # reviews, outputs [APPROVED]/[REJECTED]
    REFACTORER  = "refactorer"   # fixes rejected code
    QA          = "qa"           # verifies, mock-compiles
    CUSTOM      = "custom"       # user-defined role

class LangMode(Enum):
    ASM_C      = "asm/c"
    PYTHON     = "python"
    JAVASCRIPT = "javascript"
    ANY        = "any"

@dataclass
class AgentDef:
    """Definition of one agent in the registry."""
    agent_id: str
    name: str
    emoji: str
    color: str
    role_type: RoleType
    language: LangMode
    system_prompt: str
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    enabled: bool = True
    thinking_effort: str = "default"  # "default" | "max"

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "emoji": self.emoji,
            "color": self.color,
            "role_type": self.role_type.value,
            "language": self.language.value,
            "system_prompt": self.system_prompt,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "enabled": self.enabled,
            "thinking_effort": self.thinking_effort,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentDef":
        return cls(
            agent_id=d["agent_id"], name=d["name"], emoji=d["emoji"],
            color=d["color"], role_type=RoleType(d["role_type"]),
            language=LangMode(d.get("language", "any")),
            system_prompt=d["system_prompt"], api_key=d.get("api_key", ""),
            base_url=d.get("base_url", ""), model=d.get("model", ""),
            enabled=d.get("enabled", True),
            thinking_effort=d.get("thinking_effort", "default"),
        )


# ── Built-in agent templates ──

ARCHITECT_PROMPT = textwrap.dedent("""\
You are the **SYSTEM ARCHITECT** — a low-level x86 systems engineer.

Generate standard {language} starter source code for a minimal bootable kernel.
Produce EXACTLY three code blocks with filenames:

### boot.asm
- Real-mode x86 boot sector (512 bytes, ends with 0x55 0xAA)
- Set up a basic GDT, switch to 32-bit protected mode
- Far-jump to the kernel entry point

### kernel.c
- Minimal kernel entry in C (called from boot.asm after pmode switch)
- Clear the VGA text-mode screen (0xB8000)
- Print a banner string: "AutoClaw Kernel v1.0 - Swarm Compiled"
- Halt with an infinite loop

### linker.ld
- Linker script placing .text at 0x100000, .data and .bss after
- ENTRY(kernel_main)
- Standard x86 ELF output format
""")

CRITIC_PROMPT = textwrap.dedent("""\
You are the **CODE CRITIC** — an unforgiving systems-code reviewer.

Review EVERY line for syntax errors, memory violations, linker/ABI mismatches.
Output your detailed review with line references.
At the VERY END, output EXACTLY ONE verdict: [APPROVED] or [REJECTED].
Be precise. If REJECTED, explain exactly what to fix.
""")

REFACTORER_PROMPT = textwrap.dedent("""\
You are the **REFACTORER** — a precise code fixer.
Fix EVERY issue the critic identified. Preserve original file structure.
Output corrected files in code blocks. Do NOT introduce new features.
End with [FIXED] and a summary of changes.
""")

QA_PROMPT = textwrap.dedent("""\
You are the **QA VERIFICATION ENGINE**.
Verify file completeness, cross-file consistency, and syntax.
Mock-compile: check headers, symbols, stack conventions.
Output [VERIFIED] if all checks pass, with [WARNING] lines for non-blocking issues.
""")

PYTHON_GENERATOR_PROMPT = textwrap.dedent("""\
You are a **PYTHON BACKEND ENGINEER**. Generate production-grade Python code.
Produce complete, runnable files in code blocks tagged with filenames.
Include type hints, error handling, logging, and tests.
End with a summary of all generated files.
""")

JAVASCRIPT_GENERATOR_PROMPT = textwrap.dedent("""\
You are **Peter JS** — a JavaScript/Node.js systems engineer.

Generate production-grade JavaScript/TypeScript code. Your specialties:
- Node.js backends (Express/Fastify)
- Browser APIs & DOM manipulation
- WebSocket real-time systems
- JS build tooling (Webpack, Vite, esbuild)
- Browser automation (Puppeteer/Playwright)

Produce complete, runnable files in code blocks tagged with filenames.
Include error handling, async/await patterns, and JSDoc types.
End with a summary of all generated files and how to run them.
""")

JAVASCRIPT_CRITIC_PROMPT = textwrap.dedent("""\
You are the **JS CODE CRITIC**. Review JavaScript/TypeScript code for:
- Syntax errors, missing semicolons, unmatched braces
- Memory leaks (unremoved listeners, uncleaned intervals)
- Async/await anti-patterns (fire-and-forget, missing try/catch)
- Security issues (XSS, injection, unsafe eval)
- Node.js specific: event loop blocking, unhandled rejections

Output [APPROVED] or [REJECTED] at the end with exact line references.
""")

JAVASCRIPT_REFACTORER_PROMPT = textwrap.dedent("""\
You are the **JS REFACTORER**. Fix all critic-identified issues in the JavaScript code.
Preserve original logic. Add proper error boundaries, cleanup patterns, and type safety.
End with [FIXED] and a changes summary.
""")

# ── Default agent registry ──

def build_default_agents() -> List[AgentDef]:
    return [
        AgentDef("architect", "Architect", "🏗️", "#e74c3c",
                 RoleType.GENERATOR, LangMode.ASM_C, ARCHITECT_PROMPT,
                 base_url="https://open.bigmodel.cn/api/paas/v4", model="glm-4-plus"),
        AgentDef("critic-1", "Critic Alpha", "🔍", "#f39c12",
                 RoleType.CRITIC, LangMode.ASM_C, CRITIC_PROMPT,
                 base_url="https://api.deepseek.com/v1", model="deepseek-chat",
                 thinking_effort="max"),
        AgentDef("refactorer-1", "Refactorer", "🔧", "#3498db",
                 RoleType.REFACTORER, LangMode.ASM_C, REFACTORER_PROMPT,
                 base_url="https://api.openai.com/v1", model="gpt-4o"),
        AgentDef("qa-1", "QA Verifier", "🔬", "#2ecc71",
                 RoleType.QA, LangMode.ASM_C, QA_PROMPT,
                 base_url="https://api.minimaxi.com/v1", model="abab7-chat"),
    ]



# ═══════════════════════════════════════════════════════════════
# RATE-LIMIT RETRY UTILITY
# ═══════════════════════════════════════════════════════════════

def call_free_api_with_retry(api_url: str, headers: dict, payload: dict,
                              max_retries: int = RATE_LIMIT_MAX_RETRIES,
                              base_delay: float = RATE_LIMIT_BASE_DELAY) -> dict:
    """Calls a free frontier API and automatically sleeps if rate limits are hit.
    Uses exponential backoff: 5s → 10s → 20s → 40s → 80s."""
    delay = base_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=120)

            if response.status_code == 429:
                print(f"⚠️ Rate limit hit (attempt {attempt+1}/{max_retries}). "
                      f"Sleeping {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue

            if response.status_code == 200:
                return response.json()

            # Non-200, non-429 — still retryable
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"⚠️ API error (attempt {attempt+1}/{max_retries}): {last_error}")
            time.sleep(delay)
            delay *= 2

        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            print(f"⚠️ Timeout (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            print(f"⚠️ Connection error (attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

    raise Exception(f"❌ Failed after {max_retries} retries. Last error: {last_error}")


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════


@dataclass
class AgentLog:
    agent_id: str
    name: str
    lines: List[str] = field(default_factory=list)
    status: str = "idle"
    start_ts: float = 0.0
    end_ts: float = 0.0
    loop_index: int = 0
    verdict: str = ""
    error_msg: str = ""

    @property
    def elapsed(self) -> str:
        if self.start_ts == 0: return "—"
        return f"{time.time() - self.start_ts:.1f}s"

    @property
    def terminal_text(self) -> str:
        if self.status == "idle": return "[ waiting ]"
        if self.status == "error": return f"❌\n{self.error_msg}"
        return "\n".join(self.lines) if self.lines else "[ running ]"


@dataclass
class VirtualFile:
    name: str
    content: str
    size_bytes: int = 0
    sha256: str = ""
    status: str = "pending"

    def __post_init__(self):
        self.size_bytes = len(self.content.encode("utf-8"))
        self.sha256 = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


@dataclass
class SwarmState:
    task_id: str = ""
    task_prompt: str = ""
    agent_logs: Dict[str, AgentLog] = field(default_factory=dict)
    loop_count: int = 0
    max_loops: int = 5
    verdict: str = ""
    approved: bool = False
    virtual_fs: Dict[str, VirtualFile] = field(default_factory=dict)
    global_log: List[str] = field(default_factory=list)
    phase: str = "init"
    # Dynamic agent pipeline
    generators: List[str] = field(default_factory=list)
    critics: List[str] = field(default_factory=list)
    refactorers: List[str] = field(default_factory=list)
    qa_agents: List[str] = field(default_factory=list)

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.global_log.append(f"[{ts}] {msg}")


# ═══════════════════════════════════════════════════════════════
# SIMULATED OUTPUTS (fallback when no API keys)
# ═══════════════════════════════════════════════════════════════

SIM_ARCHITECT_OUTPUT = textwrap.dedent("""\
### boot.asm
```asm
[bits 16]
[org 0x7C00]
KERNEL_OFFSET equ 0x7E00
start:
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    mov [BOOT_DRIVE], dl
    mov bx, KERNEL_OFFSET
    mov dh, 16
    mov dl, [BOOT_DRIVE]
    call disk_load
    cli
    lgdt [gdt_descriptor]
    mov eax, cr0
    or eax, 0x1
    mov cr0, eax
    jmp CODE_SEG:init_pm
[bits 32]
init_pm:
    mov ax, DATA_SEG
    mov ds, ax
    mov ss, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ebp, 0x90000
    mov esp, ebp
    call KERNEL_OFFSET
    jmp $
disk_load:
    pusha
    mov ah, 0x02
    mov al, dh
    push dx
    mov ch, 0x00
    mov cl, 0x02
    mov dh, 0x00
    int 0x13
    pop dx
    jc disk_error
    cmp al, dh
    jne sectors_error
    popa
    ret
disk_error:
    mov si, DISK_ERROR_MSG
    call print_string
    jmp $
sectors_error:
    mov si, SECTORS_ERROR_MSG
    call print_string
    jmp $
print_string:
    pusha
    mov ah, 0x0E
.loop: lodsb; or al, al; jz .done; int 0x10; jmp .loop
.done: popa; ret
DISK_ERROR_MSG db "Disk error!", 0
SECTORS_ERROR_MSG db "Sector mismatch!", 0
BOOT_DRIVE db 0
gdt_start: dd 0x0; dd 0x0
gdt_code: dw 0xFFFF; dw 0x0; db 0x0; db 10011010b; db 11001111b; db 0x0
gdt_data: dw 0xFFFF; dw 0x0; db 0x0; db 10010010b; db 11001111b; db 0x0
gdt_end:
gdt_descriptor: dw gdt_end - gdt_start - 1; dd gdt_start
CODE_SEG equ gdt_code - gdt_start
DATA_SEG equ gdt_data - gdt_start
times 510 - ($ - $$) db 0
dw 0xAA55
```

### kernel.c
```c
#define VGA_ADDRESS 0xB8000
#define VGA_WIDTH   80
#define VGA_HEIGHT  25
typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int   uint32_t;
static volatile uint16_t *const vga_buffer = (uint16_t *)VGA_ADDRESS;
static void vga_print(const char *str, uint8_t row, uint8_t col, uint8_t attr) {
    if (row >= VGA_HEIGHT) row = VGA_HEIGHT - 1;
    if (col >= VGA_WIDTH)  col = VGA_WIDTH - 1;
    uint16_t *dest = (uint16_t *)vga_buffer + (row * VGA_WIDTH) + col;
    while (*str && col < VGA_WIDTH) {
        *dest++ = (uint16_t)(*str++) | ((uint16_t)attr << 8);
        col++;
    }
}
static void vga_clear(uint8_t attr) {
    for (uint32_t i = 0; i < VGA_WIDTH * VGA_HEIGHT; i++)
        vga_buffer[i] = (uint16_t)' ' | ((uint16_t)attr << 8);
}
void kernel_main(void) {
    vga_clear(0x0F);
    vga_print("AutoClaw Kernel v1.0 - Swarm Compiled", 12, 20, 0x0F);
    vga_print("All agents: PASS", 14, 28, 0x0A);
    while (1) { __asm__ volatile ("hlt"); }
}
```

### linker.ld
```ld
ENTRY(kernel_main)
SECTIONS {
    . = 0x100000;
    .text ALIGN(4K) : { *(.text*) }
    .rodata ALIGN(4K) : { *(.rodata*) }
    .data ALIGN(4K) : { *(.data*) }
    .bss ALIGN(4K) : { *(COMMON) *(.bss*) }
}
```
""")

SIM_CRITIC_APPROVED = textwrap.dedent("""\
## Code Review
- ✅ boot.asm: correct GDT, pmode switch, 512B with 0xAA55
- ✅ kernel.c: volatile VGA buffer, bounds-checked print, halt loop
- ✅ linker.ld: ENTRY(kernel_main), section alignment correct
[APPROVED]
""")

SIM_CRITIC_REJECTED = textwrap.dedent("""\
## Code Review
- ❌ boot.asm: KERNEL_OFFSET at 0x1000 overlaps IVT; use 0x7E00
- ❌ kernel.c: missing bounds check on vga_print column
- ❌ linker.ld: missing ALIGN directives
[REJECTED]
""")

SIM_REFACTORER_OUTPUT = textwrap.dedent("""\
## Refactored Code
### boot.asm
```asm
KERNEL_OFFSET equ 0x7E00
; (rest of boot.asm identical but with KERNEL_OFFSET fixed)
```
### kernel.c
```c
static void vga_print(const char *str, uint8_t row, uint8_t col, uint8_t attr) {
    if (row >= VGA_HEIGHT) row = VGA_HEIGHT - 1;
    if (col >= VGA_WIDTH)  col = VGA_WIDTH - 1;
    /* ... bounds-checked write ... */
}
```
### linker.ld
```ld
SECTIONS {
    .text ALIGN(4K) : { *(.text*) }
    .data ALIGN(4K) : { *(.data*) }
    .bss ALIGN(4K) : { *(COMMON) *(.bss*) }
}
```
[FIXED] — KERNEL_OFFSET corrected, bounds check added, ALIGN directives added.
""")

SIM_QA_OUTPUT = textwrap.dedent("""\
## QA Verification
| File | Status | Size |
|------|--------|------|
| boot.asm | ✅ VALID | 512B |
| kernel.c | ✅ VALID | 1.2KB |
| linker.ld | ✅ VALID | 0.3KB |
- ✅ All files present and complete
- ✅ Cross-file symbols consistent
- ✅ No undefined externals
[VERIFIED]
""")

SIM_JS_GENERATOR = textwrap.dedent("""\
## Peter JS — Generated Code
### server.js
```javascript
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// In-memory store
const tasks = new Map();
let nextId = 1;

app.get('/api/tasks', (req, res) => {
  res.json(Array.from(tasks.values()));
});

app.post('/api/tasks', (req, res) => {
  const { title, description } = req.body;
  if (!title) return res.status(400).json({ error: 'title required' });
  const task = {
    id: String(nextId++),
    title,
    description: description || '',
    status: 'pending',
    createdAt: new Date().toISOString()
  };
  tasks.set(task.id, task);
  res.status(201).json(task);
});

app.get('/api/tasks/:id', (req, res) => {
  const task = tasks.get(req.params.id);
  if (!task) return res.status(404).json({ error: 'not found' });
  res.json(task);
});

app.put('/api/tasks/:id', (req, res) => {
  const task = tasks.get(req.params.id);
  if (!task) return res.status(404).json({ error: 'not found' });
  Object.assign(task, req.body, { id: task.id, createdAt: task.createdAt });
  res.json(task);
});

app.delete('/api/tasks/:id', (req, res) => {
  if (!tasks.delete(req.params.id)) return res.status(404).json({ error: 'not found' });
  res.status(204).send();
});

// Graceful shutdown
process.on('SIGTERM', () => { server.close(() => process.exit(0)); });
const server = app.listen(PORT, () => console.log(`Peter JS server on :${PORT}`));
```

### package.json
```json
{
  "name": "peter-js-task-api",
  "version": "1.0.0",
  "main": "server.js",
  "scripts": { "start": "node server.js", "dev": "node --watch server.js" },
  "dependencies": { "express": "^4.18.2" }
}
```

### test.js
```javascript
const assert = require('assert');
async function runTests() {
  const base = 'http://localhost:3000/api';
  // Create
  const r1 = await fetch(`${base}/tasks`, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({title:'Test task'})
  });
  assert.strictEqual(r1.status, 201);
  const task = await r1.json();
  assert(task.id);
  assert.strictEqual(task.title, 'Test task');
  // Get
  const r2 = await fetch(`${base}/tasks/${task.id}`);
  assert.strictEqual(r2.status, 200);
  // Delete
  const r3 = await fetch(`${base}/tasks/${task.id}`, {method:'DELETE'});
  assert.strictEqual(r3.status, 204);
  console.log('✅ All tests passed');
}
runTests().catch(e => { console.error(e); process.exit(1); });
```
""")


# ═══════════════════════════════════════════════════════════════
# MULTI-ENDPOINT LLM CLIENT
# ═══════════════════════════════════════════════════════════════

class MultiLLMClient:
    def __init__(self):
        self.clients: Dict[str, AsyncOpenAI] = {}
        self.sim_mode: Dict[str, bool] = {}
        self.models: Dict[str, str] = {}
        self.agent_prompts: Dict[str, str] = {}
        self.agent_roles: Dict[str, RoleType] = {}
        self.agent_langs: Dict[str, LangMode] = {}

    def register_agent(self, agent: AgentDef):
        key = agent.api_key.strip()
        if not key:
            self.sim_mode[agent.agent_id] = True
        else:
            self.sim_mode[agent.agent_id] = False
            self.clients[agent.agent_id] = AsyncOpenAI(
                api_key=key,
                base_url=agent.base_url.strip() or "https://api.openai.com/v1",
            )
        self.models[agent.agent_id] = agent.model.strip() or "gpt-4"
        self.agent_prompts[agent.agent_id] = agent.system_prompt
        self.agent_roles[agent.agent_id] = agent.role_type
        self.agent_langs[agent.agent_id] = agent.language

    def is_sim(self, agent_id: str) -> bool:
        return self.sim_mode.get(agent_id, True)

    def unregister(self, agent_id: str):
        for d in [self.clients, self.sim_mode, self.models, self.agent_prompts, self.agent_roles, self.agent_langs]:
            d.pop(agent_id, None)


# ═══════════════════════════════════════════════════════════════
# SWARM ENGINE — dynamic agent pipeline
# ═══════════════════════════════════════════════════════════════

class SwarmEngine:
    def __init__(self, mllm: MultiLLMClient, agents: List[AgentDef]):
        self.mllm = mllm
        self.agents: Dict[str, AgentDef] = {a.agent_id: a for a in agents}

    # ── streaming ──

    async def _call_llm(self, agent_id: str, user_content: str,
                        extra_body: Optional[dict] = None) -> AsyncGenerator[str, None]:
        if self.mllm.is_sim(agent_id):
            for chunk in self._sim_stream(agent_id, user_content):
                yield chunk
            return

        client = self.mllm.clients[agent_id]
        model = self.mllm.models.get(agent_id, "gpt-4")
        system_prompt = self.mllm.agent_prompts.get(agent_id, "")
        agent = self.agents.get(agent_id)
        lang = (agent.language.value if agent else "any")

        messages = [
            {"role": "system", "content": system_prompt.replace("{language}", lang)},
            {"role": "user", "content": user_content},
        ]
        kwargs = {"model": model, "messages": messages,
                  "max_tokens": 8192, "stream": True}
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.RateLimitError as e:
            # Rate limit hit — retry with exponential backoff
            delay = RATE_LIMIT_BASE_DELAY
            for rtry in range(RATE_LIMIT_MAX_RETRIES):
                yield f"\n⚠️ Rate limit hit (retry {rtry+1}/{RATE_LIMIT_MAX_RETRIES}) — waiting {delay}s...\n"
                await asyncio.sleep(delay)
                delay *= 2
                try:
                    stream = await client.chat.completions.create(**kwargs)
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
                except openai.RateLimitError:
                    continue
                except Exception as e2:
                    yield f"\n\n❌ API Error [{agent_id}] after retries: {str(e2)}\n"
                    break
            yield f"\n\n❌ Rate limit persisted after {RATE_LIMIT_MAX_RETRIES} retries. Falling back to simulation...\n\n"
            for chunk in self._sim_stream(agent_id, user_content):
                yield chunk
        except Exception as e:
            yield f"\n\n❌ API Error [{agent_id}]: {str(e)}\nFalling back to simulation...\n\n"
            for chunk in self._sim_stream(agent_id, user_content):
                yield chunk

    def _sim_stream(self, agent_id: str, _user_content: str):
        agent = self.agents.get(agent_id)
        role = agent.role_type if agent else RoleType.CUSTOM
        lang = agent.language if agent else LangMode.ANY

        sims = {}
        if lang == LangMode.JAVASCRIPT:
            sims = {
                "generator": SIM_JS_GENERATOR,
                "critic": SIM_CRITIC_APPROVED,
                "refactorer": "## JS Refactorer\nAll critic issues resolved.\n[FIXED]",
                "qa": "## JS QA\n- ✅ All files complete\n- ✅ No syntax errors\n- ✅ Tests pass\n[VERIFIED]",
            }
        else:
            sims = {
                "generator": SIM_ARCHITECT_OUTPUT,
                "critic": SIM_CRITIC_APPROVED,
                "refactorer": SIM_REFACTORER_OUTPUT,
                "qa": SIM_QA_OUTPUT,
            }

        text = sims.get(role.value, "[no simulation for this role]")
        for i, ch in enumerate(text):
            yield ch
            if i % 5 == 0:
                time.sleep(0.001)

    # ── main swarm loop ──

    async def run_swarm(self, task_prompt: str, max_loops: int,
                        critic_mode: str = "strict") -> AsyncGenerator[dict, None]:
        state = SwarmState(task_prompt=task_prompt, max_loops=max_loops)
        state.task_id = f"swarm-{int(time.time())}"

        # Categorize enabled agents by role
        for aid, agent in self.agents.items():
            if not agent.enabled:
                continue
            state.agent_logs[aid] = AgentLog(agent_id=aid, name=agent.name)
            if agent.role_type == RoleType.GENERATOR:
                state.generators.append(aid)
            elif agent.role_type == RoleType.CRITIC:
                state.critics.append(aid)
            elif agent.role_type == RoleType.REFACTORER:
                state.refactorers.append(aid)
            elif agent.role_type == RoleType.QA:
                state.qa_agents.append(aid)

        state.log(f"🚀 Swarm initialized — {len(state.generators)} generators, "
                  f"{len(state.critics)} critics, {len(state.refactorers)} refactorers, "
                  f"{len(state.qa_agents)} QA, max {max_loops} loops")

        # ── Phase 1: All generators run in parallel ──
        state.phase = "generators"
        for aid in state.generators:
            async for result in self._run_single_agent(state, aid, "generating code",
                                              task_prompt):

                yield result
        self._extract_all_files(state)

        # ── Phase 2: Critic loop (max N) ──
        for loop_idx in range(1, max_loops + 1):
            state.loop_count = loop_idx
            state.phase = "critic"
            state.log(f"🔍 Critic Loop {loop_idx}/{max_loops}")

            code_snap = self._build_code_snapshot(state)
            all_approved = True
            any_verdict = False

            for aid in state.critics:
                agent = self.agents[aid]
                extra = {}
                if agent.thinking_effort == "max":
                    extra = {"thinking": {"type": "enabled"}}
                full = await self._collect_stream(
                    state, aid, f"Review this code:\n\n{code_snap}", extra)
                any_verdict = True

                if "[APPROVED]" in full:
                    state.agent_logs[aid].verdict = "APPROVED"
                elif "[REJECTED]" in full:
                    state.agent_logs[aid].verdict = "REJECTED"
                    all_approved = False
                else:
                    state.agent_logs[aid].verdict = "NONE"
                    all_approved = False

            yield self._build_ui_state(state, f"🔍 Critics done (loop {loop_idx})")

            if not any_verdict:
                state.log("⚠️ No critics active — advancing")
                state.approved = True
                break

            if all_approved:
                state.approved = True
                state.verdict = "APPROVED"
                state.log(f"✅ All critics APPROVED at loop {loop_idx}")
                yield self._build_ui_state(state,
                    f"✅ All critics APPROVED (loop {loop_idx}) → QA")
                break

            # ── Refactor if rejected and loops remain ──
            if loop_idx < max_loops:
                state.phase = "refactorer"
                for aid in state.refactorers:
                    async for result in self._run_single_agent(state, aid,
                        f"refactoring (loop {loop_idx})",
                        f"Original:\n{code_snap}\n\nFix all REJECTED issues."):
                        yield result
                self._extract_all_files(state)
                yield self._build_ui_state(state,
                    f"🔧 Refactorer done → re-submitting (loop {loop_idx+1})")

        if not state.approved:
            state.phase = "exhausted"
            state.log(f"❌ Max loops ({max_loops}) exhausted")
            yield self._build_ui_state(state, f"❌ MAX LOOPS ({max_loops}) exhausted")
            return

        # ── Phase 3: QA ──
        state.phase = "qa"
        for aid in state.qa_agents:
            code_snap = self._build_code_snapshot(state)
            async for result in self._run_single_agent(state, aid, "verifying",
                f"Verify this approved code:\n\n{code_snap}"):

                yield result

        # ── Phase 4: Mock-compile ──
        state.phase = "mock-compile"
        async for result in self._run_mock_compile(state):
            yield result

        state.phase = "complete"
        state.log("✅ Swarm complete — all stages passed")
        yield self._build_ui_state(state, "✅ SWARM COMPLETE — All agents passed")

    # ── agent runner ──

    async def _run_single_agent(self, state: SwarmState, agent_id: str,
                                 phase_label: str, prompt: str):
        log = state.agent_logs[agent_id]
        log.status = "running"
        log.start_ts = time.time()
        log.lines = []
        agent = self.agents[agent_id]
        state.log(f"{agent.emoji} {agent.name} — {phase_label}")

        yield self._build_ui_state(state, f"{agent.emoji} {agent.name} — {phase_label}...")

        buf = []
        extra = {}
        if agent.thinking_effort == "max":
            extra = {"thinking": {"type": "enabled"}}

        async for token in self._call_llm(agent_id, prompt, extra):
            buf.append(token)
            log.lines.append(token)

        log.status = "done"
        log.end_ts = time.time()
        state.log(f"{agent.emoji} {agent.name} done — {len(buf)} tokens")
        yield self._build_ui_state(state, f"{agent.emoji} {agent.name} done")

    async def _collect_stream(self, state: SwarmState, agent_id: str,
                               prompt: str, extra: Optional[dict]) -> str:
        """Collect full output from an agent without intermediate yields."""
        log = state.agent_logs[agent_id]
        log.status = "running"
        log.start_ts = time.time()
        log.lines = []
        buf = []
        async for token in self._call_llm(agent_id, prompt, extra):
            buf.append(token)
            log.lines.append(token)
        log.status = "done"
        log.end_ts = time.time()
        return "".join(buf)

    # ── virtual filesystem ──

    def _extract_all_files(self, state: SwarmState):
        for aid in state.generators + state.refactorers:
            log = state.agent_logs.get(aid)
            if not log:
                continue
            full = "\n".join(log.lines)
            self._extract_files(state, full)

    def _extract_files(self, state: SwarmState, text: str):
        pattern = re.compile(r'###\s+(\S+)\s*\n\s*```(?:\w+)?\n(.*?)```', re.DOTALL)
        for match in pattern.finditer(text):
            fname = match.group(1).strip()
            content = match.group(2).strip()
            vf = VirtualFile(name=fname, content=content)
            state.virtual_fs[fname] = vf
            state.log(f"📄 Virtual FS: {fname} ({vf.size_bytes}B)")

    def _build_code_snapshot(self, state: SwarmState) -> str:
        if not state.virtual_fs:
            return "[No code generated yet]"
        parts = []
        for fname in sorted(state.virtual_fs.keys()):
            vf = state.virtual_fs[fname]
            parts.append(f"### {fname}\n```\n{vf.content}\n```\n")
        return "\n".join(parts)

    async def _run_mock_compile(self, state: SwarmState):
        state.log("📦 Mock-compiling...")
        yield self._build_ui_state(state, "📦 Mock-compiling...")

        if not state.virtual_fs:
            state.log("⚠️ No files to compile")
            return

        for fname, vf in state.virtual_fs.items():
            if fname.endswith(".asm"):
                vf.status = "verified" if ("0xAA55" in vf.content or "AA55" in vf.content) else "corrupt"
            elif fname.endswith(".js"):
                vf.status = "verified" if ("function" in vf.content or "=>" in vf.content or "require" in vf.content or "import" in vf.content) else "corrupt"
            elif fname.endswith(".json"):
                try:
                    json.loads(vf.content)
                    vf.status = "verified"
                except json.JSONDecodeError:
                    vf.status = "corrupt"
            elif fname.endswith(".py"):
                vf.status = "verified" if ("def " in vf.content or "class " in vf.content or "import " in vf.content) else "corrupt"
            elif fname.endswith(".c") or fname.endswith(".h"):
                vf.status = "verified" if ("void " in vf.content or "int " in vf.content or "#include" in vf.content) else "corrupt"
            elif fname.endswith(".ld"):
                vf.status = "verified" if ("SECTIONS" in vf.content and "ENTRY" in vf.content) else "corrupt"
            else:
                vf.status = "verified" if len(vf.content) > 10 else "corrupt"
            state.log(f"  {'✅' if vf.status == 'verified' else '⚠️'} {fname}: {vf.status}")

        state.log("📦 Mock-compile complete")
        yield self._build_ui_state(state, "📦 Mock-compile done")

    # ── UI state builder ──

    def _build_ui_state(self, state: SwarmState, phase_msg: str) -> dict:
        return {
            "agent_outputs": self._render_agent_outputs(state),
            "bus_html": self._render_bus(state),
            "status_html": self._render_status(state, phase_msg),
            "fs_html": self._render_virtual_fs(state),
        }

    def _render_agent_outputs(self, state: SwarmState) -> dict:
        """Return a dict of agent_id -> terminal_text for Gradio updates."""
        out = {}
        for aid, log in state.agent_logs.items():
            out[aid] = log.terminal_text
        return out

    def _render_bus(self, state: SwarmState) -> str:
        rows = []
        for aid, log in state.agent_logs.items():
            agent = self.agents.get(aid)
            if not agent:
                continue
            icon = {"idle":"⏳","running":"🔄","done":"✅","error":"❌"}.get(log.status,"❓")
            rows.append(
                f'<tr><td>{agent.emoji} {agent.name}</td>'
                f'<td>{icon} {log.status}</td>'
                f'<td>{log.elapsed}</td>'
                f'<td>{log.verdict}</td></tr>'
            )

        fs_rows = ""
        for fname in sorted(state.virtual_fs.keys()):
            vf = state.virtual_fs[fname]
            fs_rows += (f'<tr><td>📄 {fname}</td><td>{vf.size_bytes}B</td>'
                        f'<td><code>{vf.sha256}</code></td><td>{vf.status}</td></tr>')

        return f"""<h3>🚦 Execution Bus — <code>{state.task_id}</code></h3>
<p>Phase: <b>{state.phase}</b> | Loop: <b>{state.loop_count}/{state.max_loops}</b> | Verdict: <b>{state.verdict}</b></p>
<table style="width:100%;border-collapse:collapse;color:#e2e8f0;font-size:0.82rem;">
<tr style="background:#1e293b;"><th>Agent</th><th>Status</th><th>Time</th><th>Verdict</th></tr>
{''.join(rows)}
</table>
<br/><h4>💾 Virtual FS</h4>
<table style="width:100%;border-collapse:collapse;color:#e2e8f0;font-size:0.78rem;">
<tr style="background:#1e293b;"><th>File</th><th>Size</th><th>SHA256</th><th>Status</th></tr>
{fs_rows if fs_rows else '<tr><td colspan="4">No files</td></tr>'}
</table>
<br/><h4>📜 Log</h4>
<pre style="max-height:140px;overflow-y:auto;font-size:0.72rem;color:#94a3b8;">{chr(10).join(state.global_log[-40:])}</pre>"""

    def _render_status(self, state: SwarmState, msg: str) -> str:
        colors = {"generators":"#e74c3c","critic":"#f39c12","refactorer":"#3498db",
                  "qa":"#2ecc71","complete":"#22c55e","exhausted":"#ef4444","init":"#94a3b8",
                  "mock-compile":"#a855f7"}
        color = colors.get(state.phase, "#94a3b8")
        pulse = 'class="pulse"' if state.phase not in ("complete","exhausted","init") else ""
        return f'<div style="color:{color};text-align:center;padding:10px;font-weight:700;font-size:1.05rem;" {pulse}>{msg}</div>'

    def _render_virtual_fs(self, state: SwarmState) -> str:
        if not state.virtual_fs:
            return "*No files generated*"
        parts = []
        for fname in sorted(state.virtual_fs.keys()):
            vf = state.virtual_fs[fname]
            parts.append(f"### 📄 {fname} ({vf.size_bytes}B) [{vf.status}]")
            parts.append(f"```\n{vf.content[:3000]}\n```")
            parts.append("")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
.gradio-container { max-width:100%!important; background:#0b1120!important; color:#e2e8f0!important;
    font-family:'JetBrains Mono','Fira Code',monospace!important; }
h1,h2,h3,h4 { color:#f1f5f9!important; }
#main-title { text-align:center;padding:1rem 0 0.25rem;background:linear-gradient(135deg,#1e293b,#0f172a);
    border-radius:16px;margin-bottom:1rem; }
#main-title h1 { font-size:1.8rem;font-weight:800;
    background:linear-gradient(135deg,#e74c3c,#f39c12,#3498db,#2ecc71,#a855f7);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.terminal-box textarea { background:#060a14!important;color:#00ff88!important;
    font-family:'JetBrains Mono',monospace!important;font-size:0.68rem!important;line-height:1.35!important;
    border:1px solid #1e293b!important;border-radius:8px!important;min-height:180px!important; }
.agent-header { font-weight:700;font-size:0.85rem;padding:6px 12px;border-radius:8px 8px 0 0;margin-bottom:-4px; }
#run-btn { background:linear-gradient(135deg,#e74c3c,#f39c12)!important;border:none!important;
    color:white!important;font-weight:700!important;font-size:1.05rem!important;
    padding:12px 28px!important;border-radius:12px!important;width:100%!important; }
#run-btn:hover { transform:scale(1.02)!important; }
#bus-panel { background:#0f172a!important;border:2px solid #1e293b!important;border-radius:12px!important;padding:16px!important; }
#fs-output { background:#060a14!important;color:#cbd5e1!important;font-size:0.78rem!important; }
.add-btn { background:#22c55e!important;color:white!important;font-weight:700!important;border:none!important;
    border-radius:8px!important; }
.del-btn { background:#ef4444!important;color:white!important;border:none!important;border-radius:8px!important; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
.pulse { animation:pulse 1.5s infinite; }
"""

# Agent registry persisted in Gradio State
AGENT_REGISTRY: List[AgentDef] = build_default_agents()
MLLM = MultiLLMClient()
for a in AGENT_REGISTRY:
    MLLM.register_agent(a)


def build_agent_edit_ui() -> List[gr.components.Component]:
    """Return the UI components for editing one agent in the list."""
    name = gr.Textbox(label="Name", scale=2)
    emoji = gr.Textbox(label="Emoji", value="🤖", scale=1)
    color = gr.ColorPicker(label="Color", value="#6366f1", scale=1)
    role_type = gr.Dropdown(label="Role",
        choices=["generator","critic","refactorer","qa","custom"],
        value="generator", scale=1)
    language = gr.Dropdown(label="Language",
        choices=["asm/c","python","javascript","any"],
        value="any", scale=1)
    system_prompt = gr.Textbox(label="System Prompt", lines=4)
    api_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
    base_url = gr.Textbox(label="Base URL", placeholder="https://api.openai.com/v1")
    model = gr.Textbox(label="Model", placeholder="gpt-4")
    thinking = gr.Radio(label="Thinking Effort",
        choices=["default","max"], value="default")
    return [name, emoji, color, role_type, language, system_prompt, api_key, base_url, model, thinking]


def create_ui():
    with gr.Blocks(css=CUSTOM_CSS,
                   theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
                   title="AutoClaw Unlimited Swarm Orchestrator") as demo:

        registry_state = gr.State(AGENT_REGISTRY)

        gr.HTML("""<div id="main-title">
            <h1>🎛️ Unlimited Multi-Agent Swarm Orchestrator</h1>
            <p style="color:#94a3b8;font-size:0.85rem;">
            ∞ Agents &nbsp;|&nbsp; ∞ Loops &nbsp;|&nbsp; ASM/C • Python • JavaScript (Peter JS)
            </p>
        </div>""")

        # ═══ Tab 1: Agent Registry ═══
        with gr.Tab("🔧 Agent Registry (∞ Agents)"):

            gr.Markdown("### Manage Your Swarm — Add, Edit, or Remove Agents")

            # ── Quick-add template buttons ──
            with gr.Row():
                gr.Markdown("**Quick Add:**")
            with gr.Row():
                add_asm_gen = gr.Button("➕ ASM/C Generator", size="sm", elem_classes=["add-btn"])
                add_asm_crit = gr.Button("➕ ASM/C Critic", size="sm", elem_classes=["add-btn"])
                add_asm_refac = gr.Button("➕ ASM/C Refactorer", size="sm", elem_classes=["add-btn"])
                add_asm_qa = gr.Button("➕ ASM/C QA", size="sm", elem_classes=["add-btn"])
            with gr.Row():
                add_py_gen = gr.Button("➕ Python Generator", size="sm", elem_classes=["add-btn"])
                add_js_gen = gr.Button("➕ Peter JS Generator", size="sm", elem_classes=["add-btn"])
                add_js_crit = gr.Button("➕ Peter JS Critic", size="sm", elem_classes=["add-btn"])
                add_custom = gr.Button("➕ Custom Agent", size="sm", elem_classes=["add-btn"])

            # ── Agent list display ──
            agent_list = gr.Dataframe(
                headers=["ID","Name","Role","Language","Enabled","Model"],
                datatype=["str","str","str","str","str","str"],
                interactive=False,
                label="Registered Agents",
                row_count=(10, "dynamic"),
            )

            with gr.Row():
                del_id = gr.Textbox(label="Agent ID to Remove", placeholder="architect", scale=3)
                del_btn = gr.Button("🗑️ Remove Agent", variant="stop", elem_classes=["del-btn"], scale=1)

            refresh_btn = gr.Button("🔄 Refresh Agent List")
            registry_status = gr.Textbox(label="Status", interactive=False)

            # ── Manual agent editor ──
            gr.Markdown("---\n### ✏️ Manually Add/Edit Agent")
            with gr.Row():
                edit_id = gr.Textbox(label="Agent ID", placeholder="my-agent-1", scale=2)
                edit_name = gr.Textbox(label="Name", scale=3)
                edit_emoji = gr.Textbox(label="Emoji", value="🤖", scale=1)
            with gr.Row():
                edit_color = gr.ColorPicker(label="Color", value="#6366f1")
                edit_role = gr.Dropdown(label="Role",
                    choices=["generator","critic","refactorer","qa","custom"], value="custom")
                edit_lang = gr.Dropdown(label="Language",
                    choices=["asm/c","python","javascript","any"], value="any")
                edit_thinking = gr.Radio(label="Thinking",
                    choices=["default","max"], value="default")
            edit_prompt = gr.Textbox(label="System Prompt", lines=5)
            with gr.Row():
                edit_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                edit_url = gr.Textbox(label="Base URL", placeholder="https://api.openai.com/v1")
                edit_model = gr.Textbox(label="Model", placeholder="gpt-4")
            with gr.Row():
                edit_enabled = gr.Checkbox(label="Enabled", value=True)
                save_btn = gr.Button("💾 Save Agent", variant="primary", scale=2)
                edit_status = gr.Textbox(label="Edit Status", interactive=False, scale=2)

            # ── Callbacks ──

            def refresh_agent_list(registry):
                rows = []
                for a in registry:
                    rows.append([a.agent_id, a.name, a.role_type.value,
                                 a.language.value, str(a.enabled), a.model or "—"])
                return rows

            def add_template_agent(registry, role: str, lang: str, name: str, emoji: str, color: str):
                prompts = {
                    ("generator","asm/c"): ARCHITECT_PROMPT,
                    ("critic","asm/c"): CRITIC_PROMPT,
                    ("refactorer","asm/c"): REFACTORER_PROMPT,
                    ("qa","asm/c"): QA_PROMPT,
                    ("generator","python"): PYTHON_GENERATOR_PROMPT,
                    ("generator","javascript"): JAVASCRIPT_GENERATOR_PROMPT,
                    ("critic","javascript"): JAVASCRIPT_CRITIC_PROMPT,
                    ("refactorer","javascript"): JAVASCRIPT_REFACTORER_PROMPT,
                    ("qa","javascript"): QA_PROMPT,
                }
                endpoints = {
                    "asm/c": ("https://open.bigmodel.cn/api/paas/v4","glm-4-plus"),
                    "python": ("https://api.openai.com/v1","gpt-4o"),
                    "javascript": ("https://api.openai.com/v1","gpt-4o"),
                }
                base, model = endpoints.get(lang, ("https://api.openai.com/v1","gpt-4"))
                prompt = prompts.get((role,lang), f"You are {name}. Generate code for {lang}.")
                new_id = f"{role}-{lang.replace('/','-')}-{uuid.uuid4().hex[:6]}"

                agent = AgentDef(
                    agent_id=new_id, name=name, emoji=emoji, color=color,
                    role_type=RoleType(role), language=LangMode(lang),
                    system_prompt=prompt, base_url=base, model=model,
                    thinking_effort="max" if role=="critic" else "default",
                )
                # Remove old agent with same role+lang if exists
                registry = [a for a in registry if not (
                    a.role_type.value == role and a.language.value == lang
                    and a.name == name)]
                registry.append(agent)
                MLLM.register_agent(agent)
                AGENT_REGISTRY[:] = registry
                return registry, refresh_agent_list(registry), f"✅ Added: {name} ({new_id})"

            def delete_agent(registry, agent_id):
                aid = agent_id.strip()
                if not aid:
                    return registry, refresh_agent_list(registry), "❌ Enter an agent ID"
                new_reg = [a for a in registry if a.agent_id != aid]
                if len(new_reg) == len(registry):
                    return registry, refresh_agent_list(registry), f"❌ ID '{aid}' not found"
                MLLM.unregister(aid)
                AGENT_REGISTRY[:] = new_reg
                return new_reg, refresh_agent_list(new_reg), f"🗑️ Removed: {aid}"

            def save_agent_manual(registry, agent_id, name, emoji, color, role, lang,
                                  prompt, key, url, model, thinking, enabled):
                if not agent_id.strip() or not name.strip():
                    return registry, refresh_agent_list(registry), "❌ ID and Name required"
                aid = agent_id.strip()
                agent = AgentDef(
                    agent_id=aid, name=name, emoji=emoji, color=color,
                    role_type=RoleType(role), language=LangMode(lang),
                    system_prompt=prompt, api_key=key, base_url=url, model=model,
                    thinking_effort=thinking, enabled=enabled,
                )
                registry = [a for a in registry if a.agent_id != aid]
                registry.append(agent)
                MLLM.register_agent(agent)
                AGENT_REGISTRY[:] = registry
                return registry, refresh_agent_list(registry), f"💾 Saved: {name} ({aid})"

            refresh_btn.click(fn=lambda r: refresh_agent_list(r),
                              inputs=[registry_state], outputs=[agent_list])
            del_btn.click(fn=delete_agent, inputs=[registry_state, del_id],
                          outputs=[registry_state, agent_list, registry_status])
            save_btn.click(fn=save_agent_manual,
                inputs=[registry_state, edit_id, edit_name, edit_emoji, edit_color,
                        edit_role, edit_lang, edit_prompt, edit_key, edit_url, edit_model,
                        edit_thinking, edit_enabled],
                outputs=[registry_state, agent_list, edit_status])

            # Template buttons
            add_asm_gen.click(fn=lambda r: add_template_agent(r, "generator","asm/c",
                "ASM/C Architect","🏗️","#e74c3c"),
                inputs=[registry_state], outputs=[registry_state, agent_list, registry_status])
            add_asm_crit.click(fn=lambda r: add_template_agent(r, "critic","asm/c",
                "ASM/C Critic","🔍","#f39c12"),
                inputs=[registry_state], outputs=[registry_state, agent_list, registry_status])
            add_asm_refac.click(fn=lambda r: add_template_agent(r, "refactorer","asm/c",
                "ASM/C Refactorer","🔧","#3498db"),
                inputs=[registry_state], outputs=[registry_state, agent_list, registry_status])
            add_asm_qa.click(fn=lambda r: add_template_agent(r, "qa","asm/c",
                "ASM/C QA","🔬","#2ecc71"),
                inputs=[registry_state], outputs=[registry_state, agent_list, registry_status])
            add_py_gen.click(fn=lambda r: add_template_agent(r, "generator","python",
                "Python Dev","🐍","#06b6d4"),
                inputs=[registry_state], outputs=[registry_state, agent_list, registry_status])
            add_js_gen.click(fn=lambda r: add_template_agent(r, "generator","javascript",
                "Peter JS","🇯🇸","#f7df1e"),
                inputs=[registry_state], outputs=[registry_state, agent_list, registry_status])
            add_js_crit.click(fn=lambda r: add_template_agent(r, "critic","javascript",
                "JS Critic","🔎","#f7df1e"),
                inputs=[registry_state], outputs=[registry_state, agent_list, registry_status])
            add_custom.click(fn=lambda r: add_template_agent(r, "custom","any",
                "Custom Agent","🤖","#6366f1"),
                inputs=[registry_state], outputs=[registry_state, agent_list, registry_status])

        # ═══ Tab 2: Swarm Orchestrator ═══
        with gr.Tab("🚀 Swarm Orchestrator"):
            with gr.Row():
                task_input = gr.Textbox(
                    label="📋 Build Directive",
                    placeholder="Describe what to build...",
                    lines=2, scale=6,
                    value="Generate a minimal x86 bootable kernel: boot.asm, kernel.c, linker.ld")
                with gr.Column(scale=1, min_width=180):
                    max_loops_slider = gr.Slider(
                        label="🔄 Max Critic Loops",
                        minimum=1, maximum=100, value=5, step=1,
                        info="1-100 (higher = more debate rounds)")
                    unlimited_loops = gr.Checkbox(
                        label="♾️ Unlimited loops", value=False,
                        info="Overrides slider (capped at 9999)")
                    critic_mode = gr.Radio(
                        choices=["strict","lenient"], value="strict",
                        label="Critic Mode")
                    run_btn = gr.Button("🚀 Launch Swarm", variant="primary", elem_id="run-btn")
                    stop_btn = gr.Button("⏹️ Halt", variant="stop")

            status_html = gr.HTML('<div style="color:#94a3b8;text-align:center;padding:10px;">⏳ Awaiting launch...</div>')

            # ── Dynamic agent output grid ──
            agent_outputs = gr.HTML('<div style="color:#94a3b8;text-align:center;padding:20px;">Configure agents in the 🔧 Agent Registry tab</div>')

        # ═══ Tab 3: Execution Bus ═══
        with gr.Tab("🚦 Execution Bus"):
            bus_html = gr.HTML("""<div id="bus-panel">
                <h3>🚦 Execution Bus</h3><p style="color:#94a3b8;">No active task. Launch the swarm.</p>
            </div>""")
            gr.Markdown("### 💾 Virtual File System (Mock-Compiled Output)")
            fs_output = gr.Markdown("*No files generated*", elem_id="fs-output")

        # ═══ Tab 4: About ═══
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""## 🎛️ Unlimited Multi-Agent Swarm Orchestrator

### Architecture
```
User Prompt → ALL Generators (parallel)
              ↓
         ALL Critics (parallel) ──→ [ALL APPROVED]?
              ↓ NO                      ↓ YES
         ALL Refactorers (parallel)    ALL QA (parallel)
              ↓                            ↓
         Loop back (max N)            Mock-Compile → [VERIFIED]
```

### Features
- **∞ Agents** — Add unlimited generators, critics, refactorers, and QA agents
- **∞ Loops** — Configure 1-100 or unlimited critic-refactor loops
- **Multi-Language** — ASM/C, Python, JavaScript (Peter JS)
- **Parallel Execution** — All agents at each stage run concurrently
- **API Console** — Per-agent API keys for any OpenAI-compatible provider
- **Simulation Mode** — Works without API keys with pre-recorded outputs

### Peter JS
A specialized JavaScript/Node.js agent that generates production-grade JS code:
- Express/Fastify backends, WebSocket servers
- Browser APIs, DOM manipulation
- Build tooling (Webpack, Vite, esbuild)
- Puppeteer/Playwright automation
""")

        # ── Main swarm runner ──
        async def on_launch(registry, task, max_l, unlimited, mode):
            if not task.strip():
                yield (
                    '<div style="color:#ef4444;">❌ No task</div>',
                    '<div id="bus-panel"><p>❌ No task provided</p></div>',
                    '*No files*',
                    "❌ No task",
                )
                return

            loops = 9999 if unlimited else int(max_l)

            # Build engine from current registry
            engine = SwarmEngine(MLLM, list(registry))
            out_state = {"agent_outputs": {}, "bus_html": "", "status_html": "", "fs_html": ""}

            async for state_dict in engine.run_swarm(task.strip(), loops, mode):
                out_state = state_dict
                agent_html = _render_dynamic_agent_grid(registry, state_dict.get("agent_outputs", {}))
                yield (
                    agent_html,
                    state_dict.get("bus_html", ""),
                    state_dict.get("fs_html", "*No files*"),
                    state_dict.get("status_html", '<div style="color:#94a3b8;">Running...</div>'),
                )

            # Final yield
            agent_html = _render_dynamic_agent_grid(registry, out_state.get("agent_outputs", {}))
            yield (
                agent_html,
                out_state.get("bus_html", ""),
                out_state.get("fs_html", "*No files*"),
                out_state.get("status_html", '<div style="color:#22c55e;">✅ Complete</div>'),
            )

        run_event = run_btn.click(
            fn=on_launch,
            inputs=[registry_state, task_input, max_loops_slider, unlimited_loops, critic_mode],
            outputs=[agent_outputs, bus_html, fs_output, status_html],
        )
        stop_btn.click(fn=None, cancels=[run_event])

    return demo


def _render_dynamic_agent_grid(registry: List[AgentDef],
                                outputs: Dict[str, str]) -> str:
    """Render a responsive grid of agent terminal panels."""
    if not outputs:
        return '<div style="color:#94a3b8;text-align:center;padding:40px;">🔧 Add agents in the Agent Registry tab, then launch</div>'

    # Group agents by role
    groups = {"generator": [], "critic": [], "refactorer": [], "qa": [], "custom": []}
    for agent in registry:
        if agent.enabled and agent.agent_id in outputs:
            groups.setdefault(agent.role_type.value, []).append(agent)

    parts = []
    order = ["generator", "critic", "refactorer", "qa", "custom"]
    group_labels = {"generator":"GENERATORS","critic":"CRITICS",
                    "refactorer":"REFACTORERS","qa":"QA VERIFIERS","custom":"CUSTOM"}

    for role_key in order:
        agents_list = groups.get(role_key, [])
        if not agents_list:
            continue
        parts.append(f'<h4 style="color:#94a3b8;margin:12px 0 4px;">{group_labels[role_key]} ({len(agents_list)})</h4>')
        parts.append('<div style="display:flex;flex-wrap:wrap;gap:10px;">')
        for agent in agents_list:
            text = outputs.get(agent.agent_id, "[ waiting ]")
            # Truncate for display
            display_text = text[-3000:] if len(text) > 3000 else text
            parts.append(f'''<div style="flex:1;min-width:280px;max-width:450px;">
                <div class="agent-header" style="background:{agent.color};color:white;">
                    {agent.emoji} {agent.name}
                </div>
                <pre style="background:#060a14;color:#00ff88;padding:10px;margin:0;border-radius:0 0 8px 8px;
                font-size:0.65rem;line-height:1.3;min-height:150px;max-height:300px;overflow-y:auto;
                white-space:pre-wrap;word-break:break-word;border:1px solid #1e293b;">{_escape_html(display_text)}</pre>
            </div>''')
        parts.append('</div>')

    return "".join(parts)


def _escape_html(text: str) -> str:
    return text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    demo = create_ui()
    demo.queue(default_concurrency_limit=5, max_size=20)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

if __name__ == "__main__":
    main()
