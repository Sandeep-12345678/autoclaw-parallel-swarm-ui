#!/usr/bin/env python3
"""
verify_build.py — Programmatic validation for AutoClaw Parallel Swarm UI
======================================================================
Tests async syntax, Gradio dependencies, parallel task loops, and imports.
Exits 0 on success, non-zero on failure.
"""

import sys
import traceback

FAILURES = 0
PASSES = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASSES, FAILURES
    if condition:
        PASSES += 1
        print(f"  ✅ {name}")
    else:
        FAILURES += 1
        print(f"  ❌ {name} — {detail}")


def test_imports():
    """Test that all required packages are importable."""
    print("\n📦 [1/6] Testing imports...")
    modules = [
        ("gradio", "gradio"),
        ("aiohttp", "aiohttp"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("openai", "openai"),
        ("colorama", "colorama"),
        ("asyncio", "asyncio"),
        ("json", "json"),
        ("re", "re"),
        ("textwrap", "textwrap"),
        ("threading", "threading"),
        ("dataclasses", "dataclasses"),
        ("datetime", "datetime"),
        ("enum", "enum"),
        ("concurrent.futures", "concurrent.futures"),
        ("queue", "queue"),
        ("traceback", "traceback"),
        ("typing", "typing"),
        ("os", "os"),
        ("time", "time"),
    ]
    for name, import_path in modules:
        try:
            __import__(import_path)
            check(f"Import {name}", True)
        except ImportError as e:
            check(f"Import {name}", False, str(e))


def test_app_syntax():
    """Verify app.py has no Python syntax errors."""
    print("\n🔍 [2/6] Checking app.py syntax...")
    try:
        with open("app.py", "r") as f:
            source = f.read()
        compile(source, "app.py", "exec")
        check("app.py compiles without syntax errors", True)
    except SyntaxError as e:
        check("app.py compiles without syntax errors", False, str(e))


def test_app_structure():
    """Verify app.py contains all required components."""
    print("\n🏗️ [3/6] Checking app.py structure...")
    try:
        with open("app.py", "r") as f:
            source = f.read()
    except FileNotFoundError:
        check("app.py exists", False, "File not found")
        return

    checks = [
        ("AgentRole enum", "class AgentRole(Enum)"),
        ("AGENT_CONFIGS dict", "AGENT_CONFIGS"),
        ("ExecutionBus class", "class ExecutionBus"),
        ("LLMStreamer class", "class LLMStreamer"),
        ("stream_all_agents method", "stream_all_agents"),
        ("asyncio.gather call", "asyncio.gather"),
        ("Gradio Blocks UI", "gr.Blocks"),
        ("Architect output", "architect_output"),
        ("Backend output", "backend_output"),
        ("Frontend output", "frontend_output"),
        ("Execution bus output", "bus_output"),
        ("Custom CSS", "CUSTOM_CSS"),
        ("main function", "def main()"),
        ("Launch on port 7860", "server_port=7860"),
        ("Dockerfile", "Dockerfile exists"),
    ]

    for name, pattern in checks:
        if name == "Dockerfile":
            try:
                with open("Dockerfile", "r") as f:
                    df = f.read()
                check("Dockerfile exists and has content", len(df) > 50)
            except FileNotFoundError:
                check("Dockerfile exists", False, "Not found")
        else:
            check(f"Contains: {name}", pattern in source)


def test_requirements():
    """Verify requirements.txt has all needed deps."""
    print("\n📋 [4/6] Checking requirements.txt...")
    try:
        with open("requirements.txt", "r") as f:
            reqs = f.read()
    except FileNotFoundError:
        check("requirements.txt exists", False, "Not found")
        return

    required = ["gradio", "aiohttp", "uvicorn"]
    for pkg in required:
        check(f"requirements.txt contains {pkg}", pkg.lower() in reqs.lower())


def test_async_patterns():
    """Verify proper async/await patterns in app.py."""
    print("\n⚡ [5/6] Checking async patterns...")
    try:
        with open("app.py", "r") as f:
            source = f.read()
    except FileNotFoundError:
        return

    patterns = [
        ("async def", "Has async functions"),
        ("await", "Uses await"),
        ("asyncio.create_task", "Creates async tasks"),
        ("asyncio.Queue", "Uses async queues"),
        ("asyncio.wait_for", "Uses timeout pattern"),
        ("AsyncGenerator", "Uses async generators"),
        ("yield", "Uses yield for streaming"),
    ]
    for pattern, desc in patterns:
        check(desc, pattern in source)


def test_dockerfile():
    """Verify Dockerfile is correct."""
    print("\n🐳 [6/6] Checking Dockerfile...")
    try:
        with open("Dockerfile", "r") as f:
            df = f.read()
    except FileNotFoundError:
        check("Dockerfile exists", False, "Not found")
        return

    checks = [
        ("FROM python", "FROM python" in df),
        ("Exposes port 7860", "EXPOSE 7860" in df or "7860" in df),
        ("Copies app code", "COPY . ." in df or "COPY" in df),
        ("Runs app.py", "app.py" in df),
    ]
    for name, condition in checks:
        check(name, condition)


def test_readme():
    """Quick README check."""
    try:
        with open("README.md", "r") as f:
            readme = f.read()
        check("README.md has YAML header", readme.startswith("---"))
        check("README.md has emoji config", "emoji:" in readme)
        check("README.md has sdk: docker", "sdk: docker" in readme)
        check("README.md has port 7860", "7860" in readme)
        check("README.md has GH link", "github.com/Sandeep-12345678/autoclaw-parallel-swarm-ui" in readme)
        check("README.md has HF link", "huggingface.co/spaces/sandeep-73/autoclaw-parallel-swarm-ui" in readme)
    except FileNotFoundError:
        check("README.md exists", False, "Not found")


# ── Run all tests ──

if __name__ == "__main__":
    print("=" * 60)
    print("  AutoClaw Parallel Swarm UI — Build Verification")
    print("=" * 60)

    test_imports()
    test_app_syntax()
    test_app_structure()
    test_requirements()
    test_async_patterns()
    test_dockerfile()
    test_readme()

    print("\n" + "=" * 60)
    print(f"  Results: {PASSES} passed, {FAILURES} failed")
    print("=" * 60)

    if FAILURES > 0:
        print("\n❌ VERIFICATION FAILED — See errors above.")
        sys.exit(1)
    else:
        print("\n✅ ALL CHECKS PASSED — Build is clean!")
        sys.exit(0)
