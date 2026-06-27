#!/usr/bin/env python3
"""
verify_orchestrator.py — Syntax & structure validation for
  Autonomous Multi-Agent Code-Debate Orchestrator
==============================================================
Exits 0 on clean pass, non-zero on failure.
"""

import sys
import traceback

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


# ── 1. Imports ──
print("\n📦 [1/6] Testing imports...")
modules = [
    "asyncio", "hashlib", "json", "os", "re", "sys",
    "textwrap", "time", "traceback", "dataclasses",
    "datetime", "enum", "typing", "gradio", "openai",
]
for mod in modules:
    try:
        __import__(mod)
        check(f"import {mod}", True)
    except ImportError as e:
        check(f"import {mod}", False, str(e))


# ── 2. Syntax ──
print("\n🔍 [2/6] Checking app.py syntax...")
try:
    with open("app.py") as f:
        src = f.read()
    compile(src, "app.py", "exec")
    check("app.py compiles", True)
except SyntaxError as e:
    check("app.py compiles", False, f"Line {e.lineno}: {e.msg}")


# ── 3. Structure — required components ──
print("\n🏗️  [3/6] Checking app.py structure...")
try:
    with open("app.py") as f:
        src = f.read()
except FileNotFoundError:
    print("  ❌ app.py not found")
    sys.exit(1)

required = [
    # Classes
    ("AgentRole enum",          "class AgentRole"),
    ("AgentLog dataclass",      "class AgentLog"),
    ("VirtualFile dataclass",   "class VirtualFile"),
    ("SwarmState dataclass",    "class SwarmState"),
    ("MultiLLMClient class",    "class MultiLLMClient"),
    ("SwarmEngine class",       "class SwarmEngine"),
    # Agent prompts
    ("Architect prompt",        "ARCHITECT_PROMPT"),
    ("Critic prompt",           "CRITIC_PROMPT"),
    ("Refactorer prompt",       "REFACTORER_PROMPT"),
    ("QA prompt",               "QA_PROMPT"),
    # Loop mechanics
    ("MAX_CRITIC_LOOPS = 5",    "MAX_CRITIC_LOOPS = 5"),
    ("critic_loop method",      "def _critic_loop"),
    ("run_architect method",    "def _run_architect"),
    ("run_qa method",           "def _run_qa"),
    ("run_mock_compile method", "def _run_mock_compile"),
    ("extract_files method",    "def _extract_files"),
    # Verdict parsing
    ("[APPROVED] parsing",      '"[APPROVED]" in'),
    ("[REJECTED] parsing",      '"[REJECTED]" in'),
    ("[VERIFIED] parsing",      '"[VERIFIED]" in'),
    ("[FIXED] parsing",         '[FIXED]'),
    # Virtual FS
    ("VirtualFile instantiation","VirtualFile("),
    ("boot.asm in checks",       '"boot.asm"'),
    ("kernel.c in checks",       '"kernel.c"'),
    ("linker.ld in checks",      '"linker.ld"'),
    # API config console
    ("API config tab",           'gr.Tab("🔑 API Configuration")'),
    ("Arch key field",           "arch_key"),
    ("Crit key field",           "crit_key"),
    ("Refac key field",          "refac_key"),
    ("QA key field",             "qa_key"),
    ("apply_config function",    "def apply_config"),
    # Gradio UI
    ("Swarm Orchestrator tab",   'gr.Tab("🚀 Swarm Orchestrator")'),
    ("Execution Bus tab",        'gr.Tab("🚦 Execution Bus")'),
    ("Launch event handler",     "def on_launch"),
    ("Gradio Blocks",            "gr.Blocks"),
    ("server_port=7860",         "server_port=7860"),
    # Simulation fallbacks
    ("SIM_ARCHITECT_OUTPUT",     "SIM_ARCHITECT_OUTPUT"),
    ("SIM_CRITIC_APPROVED",      "SIM_CRITIC_APPROVED"),
    ("SIM_CRITIC_REJECTED",      "SIM_CRITIC_REJECTED"),
    ("SIM_REFACTORER_OUTPUT",    "SIM_REFACTORER_OUTPUT"),
    ("SIM_QA_OUTPUT",            "SIM_QA_OUTPUT"),
    ("Simulation stream method", "def _sim_stream"),
    # Async patterns
    ("async def",                "async def"),
    ("asyncio patterns",         "AsyncGenerator"),
    ("yield for streaming",      "yield"),
]

for label, pattern in required:
    check(label, pattern in src)


# ── 4. Requirements ──
print("\n📋 [4/6] Checking requirements.txt...")
try:
    with open("requirements.txt") as f:
        reqs = f.read().lower()
except FileNotFoundError:
    check("requirements.txt exists", False, "Not found")
    reqs = ""

for pkg in ["gradio", "requests", "aiohttp", "openai"]:
    check(f"requirements: {pkg}", pkg in reqs)


# ── 5. Dockerfile ──
print("\n🐳 [5/6] Checking Dockerfile...")
try:
    with open("Dockerfile") as f:
        df = f.read()
except FileNotFoundError:
    check("Dockerfile exists", False, "Not found")
    df = ""

for item in ["FROM python", "EXPOSE 7860", "app.py"]:
    check(f"Dockerfile: {item}", item in df)


# ── 6. README ──
print("\n📄 [6/6] Checking README.md...")
try:
    with open("README.md") as f:
        rm = f.read()
except FileNotFoundError:
    check("README.md exists", False, "Not found")
    rm = ""

for item in ["---", "sdk: docker", "7860"]:
    check(f"README: {item}", item in rm[:500])


# ── Results ──
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")

if FAIL > 0:
    print("\n❌ VERIFICATION FAILED")
    sys.exit(1)
else:
    print("\n✅ ALL CHECKS PASSED — Orchestrator is clean!")
    sys.exit(0)
