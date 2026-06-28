---
title: AutoClaw Unlimited Swarm Orchestrator
emoji: 🎛️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🎛️ Unlimited Multi-Agent Swarm Orchestrator

Add **unlimited agents**, configure **unlimited critic loops**, and use **Peter JS** for JavaScript/Node.js code generation.

## ∞ Agents, ∞ Loops

```
User Prompt → ALL Generators (parallel)
              ↓
         ALL Critics (parallel) ──→ [ALL APPROVED]?
              ↓ NO                      ↓ YES
         ALL Refactorers (parallel)    ALL QA (parallel)
              ↓                            ↓
         Loop back (max N)            Mock-Compile → [VERIFIED]
```

## Agent Templates

| Agent | Language | Role |
|-------|----------|------|
| 🏗️ Architect | ASM/C | Generator |
| 🔍 Critic | ASM/C | Code Review |
| 🔧 Refactorer | ASM/C | Fix Rejected |
| 🔬 QA | ASM/C | Verify |
| 🐍 Python Dev | Python | Generator |
| 🇯🇸 **Peter JS** | JavaScript | Generator |
| 🔎 JS Critic | JavaScript | Review |

## Features

- **∞ Agents** — Add unlimited generators, critics, refactorers, QA via UI
- **∞ Loops** — Slider 1-100 or toggle unlimited critic-refactor loops
- **Peter JS** — Specialized JavaScript/Node.js agent for backend, browser, WebSocket
- **Parallel Execution** — All agents at each stage run concurrently
- **Multi-Provider** — Per-agent OpenAI-compatible API keys
- **Simulation Mode** — Works without API keys

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Links

- **GitHub:** [Sandeep-12345678/autoclaw-parallel-swarm-ui](https://github.com/Sandeep-12345678/autoclaw-parallel-swarm-ui)
- **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/sandeep-73/autoclaw-parallel-swarm-ui)
