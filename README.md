---
title: AutoClaw Parallel Swarm Workspace
emoji: 🎛️
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# 🔧 AutoClaw Parallel Swarm UI

A production-grade web UI wrapper for AutoClaw that orchestrates **three specialized AI sub-agents** working in parallel — Architect, Backend Developer, and Frontend Designer — with live side-by-side terminal logs and a central execution bus visualizer.

## Features

- **Parallel Agent Swarm** — 3 agents run simultaneously via `asyncio.gather`
- **Live Terminal Columns** — Side-by-side streaming output for each agent
- **Execution Bus Visualizer** — See how code merges without conflicts
- **Web UI** — Gradio-powered interface on port 7860
- **Docker Deployable** — Ready for Hugging Face Spaces

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Gradio Web UI                      │
│  ┌──────────┬──────────┬──────────┐                 │
│  │Architect │ Backend  │ Frontend │                 │
│  │  Agent   │  Agent   │  Agent   │                 │
│  └──────────┴──────────┴──────────┘                 │
│  ┌─────────────────────────────────────┐            │
│  │     Central Execution Bus           │            │
│  └─────────────────────────────────────┘            │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | LLM API key | Required |
| `OPENAI_BASE_URL` | API base URL | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Model name | `gpt-4` |
| `MAX_TOKENS` | Max response tokens | `4096` |

## Links

- **GitHub Repository:** [Sandeep-12345678/autoclaw-parallel-swarm-ui](https://github.com/Sandeep-12345678/autoclaw-parallel-swarm-ui)
- **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/sandeep-73/autoclaw-parallel-swarm-ui)
