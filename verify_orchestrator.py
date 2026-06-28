#!/usr/bin/env python3
"""verify_orchestrator.py — Validate Unlimited Swarm with Resilient Retry"""
import sys
P = F = 0
def c(n, ok, d=""):
    global P, F
    if ok: P += 1; print(f"  ✅ {n}")
    else: F += 1; print(f"  ❌ {n} — {d}")

print("📦 [1/5] Imports...")
for mod in ["asyncio","hashlib","json","os","re","sys","textwrap","time","traceback",
            "dataclasses","datetime","enum","typing","uuid","requests","gradio","openai"]:
    try: __import__(mod); c(f"import {mod}", True)
    except ImportError as e: c(f"import {mod}", False, str(e))

print("🔍 [2/5] Syntax...")
try:
    with open("app.py") as f: src = f.read()
    compile(src, "app.py", "exec"); c("app.py compiles", True)
except SyntaxError as e: c("app.py compiles", False, f"L{e.lineno}: {e.msg}")

print("🏗️  [3/5] Structure...")
for label, pat in [
    ("AgentDef dataclass","class AgentDef"),
    ("AgentDef.status field",'status: str = "available"'),
    ("RETRY_INTERVAL = 1800","RETRY_INTERVAL = 1800"),
    ("schedule_retry method","def schedule_retry"),
    ("_retry_loop method","def _retry_loop"),
    ("_try_revive_agent","def _try_revive_agent"),
    ("retry_queue dict","retry_queue"),
    ("No sim fallback comment","Simulation mode ONLY for agents with no API key"),
    ("Degraded status check",'"degraded"'),
    ("Mark degraded on failure","agent.status = \"degraded\""),
    ("call_free_api_with_retry","def call_free_api_with_retry"),
    ("RateLimitError handling","openai.RateLimitError"),
    ("Exponential backoff","delay *= 2"),
    ("Agent failure isolated","Swarm continues"),
    ("SwarmEngine class","class SwarmEngine"),
    ("Peter JS prompt","JAVASCRIPT_GENERATOR_PROMPT"),
    ("Unlimited loops","max_loops_slider"),
    ("Gradio port 7860","server_port=7860"),
]:
    c(label, pat in src)

print("📋 [4/5] Requirements...")
try:
    with open("requirements.txt") as f: reqs = f.read().lower()
except: reqs = ""
for pkg in ["gradio","requests","aiohttp","openai"]:
    c(f"req: {pkg}", pkg in reqs)

print("📄 [5/5] README...")
try:
    with open("README.md") as f: rm = f.read()
except: rm = ""
for item in ["---","sdk: docker","7860","GitHub","Hugging Face"]:
    c(f"README: {item}", item in rm)

print(f"\n{'='*60}\n  Results: {P} passed, {F} failed\n{'='*60}")
sys.exit(0 if F == 0 else 1)
