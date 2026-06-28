#!/usr/bin/env python3
"""
verify_orchestrator.py — Validate Unlimited Multi-Agent Swarm Orchestrator
Exits 0 on clean pass.
"""
import sys
PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✅ {name}")
    else: FAIL += 1; print(f"  ❌ {name} — {detail}")

print("📦 [1/5] Imports...")
for mod in ["asyncio","hashlib","json","os","re","sys","textwrap","time","traceback",
            "dataclasses","datetime","enum","typing","uuid","gradio","openai"]:
    try: __import__(mod); check(f"import {mod}", True)
    except ImportError as e: check(f"import {mod}", False, str(e))

print("🔍 [2/5] Syntax...")
try:
    with open("app.py") as f: src = f.read()
    compile(src, "app.py", "exec"); check("app.py compiles", True)
except SyntaxError as e: check("app.py compiles", False, f"Line {e.lineno}: {e.msg}")

print("🏗️  [3/5] Structure...")
checks = [
    ("AgentDef dataclass","class AgentDef"),
    ("RoleType enum","class RoleType"),
    ("LangMode enum","class LangMode"),
    ("SwarmEngine class","class SwarmEngine"),
    ("MultiLLMClient class","class MultiLLMClient"),
    ("Dynamic agent registry","build_default_agents"),
    ("Unlimited loops slider","max_loops_slider"),
    ("Unlimited checkbox","unlimited_loops"),
    ("Peter JS prompt","JAVASCRIPT_GENERATOR_PROMPT"),
    ("JS Critic prompt","JAVASCRIPT_CRITIC_PROMPT"),
    ("JS Refactorer prompt","JAVASCRIPT_REFACTORER_PROMPT"),
    ("Agent Registry tab","Agent Registry"),
    ("Add template button","add_template_agent"),
    ("Delete agent","delete_agent"),
    ("Save agent manual","save_agent_manual"),
    ("Parallel generators","state.generators"),
    ("Parallel critics","state.critics"),
    ("Parallel refactorers","state.refactorers"),
    ("Parallel QA","state.qa_agents"),
    ("Swarm runner","run_swarm"),
    ("Mock compile","_run_mock_compile"),
    ("Dynamic grid render","_render_dynamic_agent_grid"),
    ("[APPROVED] parsing",'"[APPROVED]" in'),
    ("[REJECTED] parsing",'"[REJECTED]" in'),
    ("[VERIFIED] parsing",'[VERIFIED]'),
    ("[FIXED] parsing",'[FIXED]'),
    ("SIM_JS_GENERATOR","SIM_JS_GENERATOR"),
    ("Gradio port 7860","server_port=7860"),
    ("Gradio Blocks","gr.Blocks"),
    ("Gradio State registry","gr.State"),
    ("Server launch","def main()"),
]
for label, pat in checks:
    check(label, pat in src)

print("📋 [4/5] Requirements...")
try:
    with open("requirements.txt") as f: reqs = f.read().lower()
except FileNotFoundError: reqs = ""
for pkg in ["gradio","requests","aiohttp","openai"]:
    check(f"req: {pkg}", pkg in reqs)

print("📄 [5/5] README...")
try:
    with open("README.md") as f: rm = f.read()
except FileNotFoundError: rm = ""
for item in ["---","sdk: docker","7860","GitHub","Hugging Face"]:
    check(f"README: {item}", item in rm)

print(f"\n{'='*60}\n  Results: {PASS} passed, {FAIL} failed\n{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
