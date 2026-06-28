#!/usr/bin/env python3
"""verify_orchestrator.py — Validate improved swarm orchestrator"""
import sys
P = F = 0
def c(n, ok, d=""):
    global P, F
    if ok: P += 1; print(f"  ✅ {n}")
    else: F += 1; print(f"  ❌ {n} — {d}")

print("📦 [1/5] Imports...")
for mod in ["asyncio","hashlib","json","os","re","sys","textwrap","time","traceback",
            "dataclasses","datetime","enum","typing","uuid","requests","io","zipfile",
            "gradio","openai"]:
    try: __import__(mod); c(f"import {mod}", True)
    except ImportError as e: c(f"import {mod}", False, str(e))

print("🔍 [2/5] Syntax...")
try:
    with open("app.py") as f: src = f.read()
    compile(src, "app.py", "exec"); c("app.py compiles", True)
except SyntaxError as e: c("app.py compiles", False, f"L{e.lineno}: {e.msg}")

print("🏗️  [3/5] Structure...")
checks = [
    ("True parallel generators", "asyncio.gather"),
    ("Parallel critics", "critic_tasks"),
    ("Parallel refactorers", "refac_tasks"),
    ("Parallel QA", "qa_tasks"),
    ("Agent health status", 'status == "available"'),
    ("RETRY_INTERVAL", "RETRY_INTERVAL"),
    ("Retry loop", "_retry_loop"),
    ("ensure_retry_loop", "ensure_retry_loop"),
    ("Health dashboard tab", "💓 Health"),
    ("Download button", "download_btn"),
    ("Build download fn", "def build_download"),
    ("API timeout", "timeout"),
    ("call_free_api_with_retry", "call_free_api_with_retry"),
    ("No sim fallback", "Simulation mode ONLY"),
    ("Rate limit handler", "RateLimitError"),
    ("Peter JS", "JAVASCRIPT_GENERATOR_PROMPT"),
    ("Agent Registry tab", "Agent Registry"),
    ("Swarm Orchestrator tab", "Swarm Orchestrator"),
    ("Unlimited loops", "unlimited_loops"),
    ("Port 7860", "server_port=7860"),
]
for label, pat in checks:
    c(label, pat in src)

print("📋 [4/5] Requirements...")
try:
    with open("requirements.txt") as f: reqs = f.read().lower()
except: reqs = ""
for pkg in ["gradio","requests","aiohttp","openai"]:
    c(f"req: {pkg}", pkg in reqs)

print("📄 [5/5] Config...")
try:
    with open("README.md") as f: rm = f.read()
except: rm = ""
for item in ["---","sdk: docker","7860"]:
    c(f"README: {item}", item in rm[:500])

print(f"\n{'='*60}\n  Results: {P} passed, {F} failed\n{'='*60}")
sys.exit(0 if F == 0 else 1)
