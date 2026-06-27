---
title: AutoClaw Multi-Agent Code Orchestrator
emoji: 🎛️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🎛️ Autonomous Multi-Agent Code-Debate Orchestrator

A Gradio web UI that orchestrates 4 specialized AI agents in an autonomous debate loop to generate, critique, refactor, and verify **x86 kernel source code**.

## Swarm Architecture

```
User → Architect (GLM-5.2) → Critic (DeepSeek-V4-Pro)
                                   ↓
                    ┌── [APPROVED] ──→ QA (MiniMax-M3) → [VERIFIED]
                    │
                    └── [REJECTED] → Refactorer (Ornith-397B) → Critic (max 5 loops)
```

## Agents

| # | Agent | Provider | Role |
|---|-------|----------|------|
| 🏗️ | Architect | GLM-5.2 / Zhipu | Generates boot.asm, kernel.c, linker.ld |
| 🔍 | Critic | DeepSeek-V4-Pro | Reviews syntax, memory, ABI with `thinking_effort=max` |
| 🔧 | Refactorer | Ornith-1.0-397B | Fixes rejected code, resubmits to Critic (max 5 loops) |
| 🔬 | QA | MiniMax-M3 | Mock-compiles, verifies completeness, issues [VERIFIED] |

## Features

- **API Configuration Console** — Per-agent API keys & endpoint management
- **Live Split-Column Terminal Logs** — All 4 agents stream in real-time side-by-side
- **Execution Bus** — Central state visualizer with agent status, loop count, verdict tracking
- **Virtual File System** — Mock-compiled file array (boot.asm, kernel.c, linker.ld) with SHA256
- **Simulation Mode** — Full demo with pre-recorded x86 code when API keys are not provided

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` — configure API keys in the 🔑 tab, then launch from 🚀 tab.

## Links

- **GitHub:** [Sandeep-12345678/autoclaw-parallel-swarm-ui](https://github.com/Sandeep-12345678/autoclaw-parallel-swarm-ui)
- **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/sandeep-73/autoclaw-parallel-swarm-ui)
