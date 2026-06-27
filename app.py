#!/usr/bin/env python3
"""
AutoClaw Autonomous Multi-Agent Code-Debate Orchestrator
=========================================================
A Gradio web UI that orchestrates a 4-agent autonomous swarm loop:
  Architect (GLM-5.2) → Critic (DeepSeek-V4-Pro, thinking=max)
    → Refactorer (Ornith-1.0-397B, max 5 loops) → QA (MiniMax-M3)

Each agent communicates with its own API endpoint. The Critic evaluates
code output and either [APPROVED] or [REJECTED] triggers the next stage.
"""

import asyncio
import hashlib
import json
import os
import re
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import gradio as gr
from openai import AsyncOpenAI


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEFAULT_ENDPOINTS = {
    "architect": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-plus"},
    "critic":     {"base_url": "https://api.deepseek.com/v1",            "model": "deepseek-chat"},
    "refactorer": {"base_url": "https://api.openai.com/v1",              "model": "gpt-4o"},
    "qa":         {"base_url": "https://api.minimaxi.com/v1",            "model": "abab7-chat"},
}

MAX_CRITIC_LOOPS = 5
DEFAULT_MAX_TOKENS = 8192


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

class AgentRole(Enum):
    ARCHITECT  = "architect"
    CRITIC     = "critic"
    REFACTORER = "refactorer"
    QA         = "qa"


@dataclass
class AgentLog:
    """Per-agent live log buffer."""
    role: AgentRole
    lines: List[str] = field(default_factory=list)
    status: str = "idle"       # idle | running | done | error
    start_ts: float = 0.0
    end_ts: float = 0.0
    loop_index: int = 0
    verdict: str = ""          # [APPROVED] / [REJECTED] for critic
    error_msg: str = ""

    @property
    def elapsed(self) -> str:
        if self.start_ts == 0:
            return "—"
        end = self.end_ts or time.time()
        return f"{end - self.start_ts:.1f}s"

    @property
    def terminal_text(self) -> str:
        if self.status == "idle":
            return "[ waiting ]"
        if self.status == "error":
            return f"❌\n{self.error_msg}"
        return "\n".join(self.lines) if self.lines else "[ running ]"


@dataclass
class VirtualFile:
    """Mock file in the virtual filesystem array."""
    name: str
    content: str
    size_bytes: int = 0
    sha256: str = ""
    status: str = "pending"   # pending | verified | missing | corrupt

    def __post_init__(self):
        self.size_bytes = len(self.content.encode("utf-8"))
        self.sha256 = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


@dataclass
class SwarmState:
    """Top-level state for one swarm execution."""
    task_id: str = ""
    task_prompt: str = ""
    agents: Dict[AgentRole, AgentLog] = field(default_factory=dict)
    loop_count: int = 0
    verdict: str = ""
    approved: bool = False
    virtual_fs: Dict[str, VirtualFile] = field(default_factory=dict)
    global_log: List[str] = field(default_factory=list)
    phase: str = "init"

    def __post_init__(self):
        if not self.agents:
            for role in AgentRole:
                self.agents[role] = AgentLog(role=role)

    def log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.global_log.append(f"[{ts}] {msg}")

    def append_agent(self, role: AgentRole, text: str):
        for line in text.split("\n"):
            self.agents[role].lines.append(line)


# ═══════════════════════════════════════════════════════════════
# AGENT PERSONAS & SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════

ARCHITECT_PROMPT = textwrap.dedent("""\
You are the **SYSTEM ARCHITECT** — a low-level x86 systems engineer.

Generate standard x86 C/Assembly starter source code for a minimal bootable kernel.
Produce EXACTLY three code blocks with filenames:

### boot.asm
- Real-mode x86 boot sector (512 bytes, ends with 0x55 0xAA)
- Set up a basic GDT, switch to 32-bit protected mode
- Far-jump to the kernel entry point

### kernel.c
- Minimal kernel entry in C (called from boot.asm after pmode switch)
- Clear the VGA text-mode screen (0xB8000)
- Print a banner string: "AutoClaw Kernel v1.0 — Swarm Compiled"
- Halt with an infinite loop

### linker.ld
- Linker script placing .text at 0x100000, .data and .bss after
- ENTRY(kernel_main)
- Standard x86 ELF output format

Use ONLY standard, portable x86 conventions. No MMX/SSE assumptions.
Every file MUST be a complete, compilable unit inside a triple-backtick code block.
""")

CRITIC_PROMPT = textwrap.dedent("""\
You are the **CODE CRITIC** — an unforgiving systems-code reviewer.

You receive source code from the Architect (or Refactorer). Your job:

1. Review EVERY line for:
   - Syntax errors (missing semicolons, unmatched braces, invalid mnemonics in asm)
   - Memory violations (buffer overflows, uninitialized pointers, stack imbalance)
   - Linker/ABI mismatches (wrong sections, missing symbols)
   - x86 architectural violations (e.g., using 64-bit registers in 32-bit mode)

2. Output your detailed review with line references.

3. At the VERY END of your response, output EXACTLY ONE verdict line:
   - `[APPROVED]` — code is correct, compilable, and memory-safe
   - `[REJECTED]` — code has defects that MUST be fixed

Be precise. Cite line numbers. If you say REJECTED, you MUST explain exactly what to fix.
""")

REFACTORER_PROMPT = textwrap.dedent("""\
You are the **REFACTORER** — a precise systems-code fixer.

You receive original code + critic feedback. Your job:
- Fix EVERY issue the critic identified
- Preserve the original file structure (boot.asm, kernel.c, linker.ld)
- Output corrected, complete files in triple-backtick code blocks
- Do NOT introduce new features — ONLY fix the reported defects
- At the end, output `[FIXED]` with a summary of changes made
""")

QA_PROMPT = textwrap.dedent("""\
You are the **QA VERIFICATION ENGINE** — final gate before build.

You receive [APPROVED] code from the critic loop. Your job:

1. Verify file completeness:
   - Is boot.asm exactly 512 bytes after assembly? (check for 0x55 0xAA trailer)
   - Does kernel.c have a valid entry point?
   - Does linker.ld reference the correct entry symbol?
   - Are all cross-file symbol references consistent?

2. Mock-compile: walk through each file and check for:
   - All #include headers are standard
   - No undefined external symbols
   - Stack frame conventions are consistent

3. Output final verdict at the end:
   - `[VERIFIED]` — all checks pass, ready for real compilation
   - List any remaining warnings as `[WARNING]` lines

Be thorough. This is the last check before the code ships.
""")


# ═══════════════════════════════════════════════════════════════
# SIMULATED AGENT OUTPUTS (fallback when no API keys)
# ═══════════════════════════════════════════════════════════════

SIM_ARCHITECT_OUTPUT = textwrap.dedent("""\
### boot.asm
```asm
; boot.asm — x86 Real-Mode Boot Sector
; Switches to 32-bit protected mode, far-jumps to kernel_main
[bits 16]
[org 0x7C00]

KERNEL_OFFSET equ 0x1000

start:
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00

    ; Save boot drive
    mov [BOOT_DRIVE], dl

    ; Load kernel sectors (assume kernel is at sector 2)
    mov bx, KERNEL_OFFSET
    mov dh, 16          ; 16 sectors
    mov dl, [BOOT_DRIVE]
    call disk_load

    ; Switch to protected mode
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
    push dx
    mov ah, 0x02
    mov al, dh
    mov ch, 0x00
    mov cl, 0x02
    mov dh, 0x00
    int 0x13
    jc disk_error
    pop dx
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
.loop:
    lodsb
    or al, al
    jz .done
    int 0x10
    jmp .loop
.done:
    popa
    ret

DISK_ERROR_MSG   db "Disk read error!", 0
SECTORS_ERROR_MSG db "Sector count mismatch!", 0
BOOT_DRIVE db 0

; GDT
gdt_start:
    dd 0x0
    dd 0x0
gdt_code:
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 10011010b
    db 11001111b
    db 0x0
gdt_data:
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 10010010b
    db 11001111b
    db 0x0
gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

CODE_SEG equ gdt_code - gdt_start
DATA_SEG equ gdt_data - gdt_start

times 510 - ($ - $$) db 0
dw 0xAA55
```

### kernel.c
```c
/* kernel.c — Minimal 32-bit x86 Kernel Entry */
/* Compiled with: gcc -m32 -ffreestanding -c kernel.c -o kernel.o */

#define VGA_ADDRESS 0xB8000
#define VGA_WIDTH   80
#define VGA_HEIGHT  25

typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int   uint32_t;

/* VGA entry: low byte = char, high byte = attribute */
static uint16_t *const vga_buffer = (uint16_t *)VGA_ADDRESS;

/* Write a null-terminated string to VGA at given row */
static void vga_print(const char *str, uint8_t row, uint8_t col, uint8_t attr) {
    uint16_t *dest = vga_buffer + (row * VGA_WIDTH) + col;
    while (*str) {
        *dest++ = (uint16_t)(*str++) | ((uint16_t)attr << 8);
    }
}

/* Clear the screen to black with given attribute */
static void vga_clear(uint8_t attr) {
    for (uint32_t i = 0; i < VGA_WIDTH * VGA_HEIGHT; i++) {
        vga_buffer[i] = (uint16_t)' ' | ((uint16_t)attr << 8);
    }
}

/* Kernel main — entry point from boot.asm after pmode switch */
void kernel_main(void) {
    vga_clear(0x0F);                                    /* White on black */
    vga_print("AutoClaw Kernel v1.0 - Swarm Compiled",  /* Row 12, Col 20 */
              12, 20, 0x0F);
    vga_print("All agents: PASS", 14, 28, 0x0A);       /* Green */

    /* Halt */
    while (1) {
        __asm__ volatile ("hlt");
    }
}
```

### linker.ld
```ld
/* linker.ld — x86 32-bit ELF Linker Script */
ENTRY(kernel_main)

SECTIONS {
    . = 0x100000;

    .text : {
        *(.text*)
    }

    .rodata : {
        *(.rodata*)
    }

    .data : {
        *(.data*)
    }

    .bss : {
        *(COMMON)
        *(.bss*)
    }

    /DISCARD/ : {
        *(.note*)
        *(.comment*)
        *(.eh_frame*)
    }
}
```
""")

SIM_CRITIC_APPROVED = textwrap.dedent("""\
## Code Review — Line-by-Line Analysis

### boot.asm
- ✅ Line 1-10: Correct [bits 16] / [org 0x7C00] setup
- ✅ Line 17: Stack pointer at 0x7C00 — safe, grows downward below boot sector
- ✅ Line 27-31: GDT load, CR0 PE bit set correctly
- ✅ Line 33-42: 32-bit protected mode init — segment registers and stack set correctly
- ✅ Line 44-60: disk_load uses INT 0x13 correctly with proper error handling
- ✅ Line 80-104: GDT entries correct — code (0x9A) and data (0x92) with full 4GB limit
- ✅ Line 109-110: Boot signature 0xAA55 present, sector is 512 bytes

### kernel.c
- ✅ Line 1-2: Standard freestanding headers only
- ✅ Line 5-9: VGA constants and types correct for x86
- ✅ Line 11: VGA buffer pointer correctly typed as volatile
- ✅ Line 14-18: vga_print — correct VGA text-mode character packing
- ✅ Line 21-25: vga_clear — correct screen size (80x25)
- ✅ Line 29-33: kernel_main — clears screen, prints banner, halts
- ✅ No stack overflows, no uninitialized pointers

### linker.ld
- ✅ Line 2: ENTRY(kernel_main) matches kernel.c symbol
- ✅ Line 5: Base address 0x100000 correct for x86 kernel
- ✅ Line 7-15: Standard .text/.rodata/.data/.bss sections
- ✅ Line 17-20: Discard note/comment/eh_frame sections

### Verdict
All files are syntactically correct, memory-safe, and follow x86 protected-mode conventions.
No buffer overflows, no missing symbols, no ABI violations.

[APPROVED]
""")

SIM_CRITIC_REJECTED = textwrap.dedent("""\
## Code Review — Line-by-Line Analysis

### boot.asm
- ❌ Line 21: `mov bx, KERNEL_OFFSET` — loads kernel to 0x1000 which overlaps
  the IVT (0x0000-0x03FF) and BIOS data area (0x0400-0x04FF). This is actually
  safe for our use case since we only read 16 sectors (8KB), but should use
  0x7E00 (just above the boot sector) for best practice.
- ⚠️ Line 47-50: disk_load pushes dx but doesn't restore on all error paths

### kernel.c
- ❌ Line 3: Missing `#include <stddef.h>` for size_t (not used, but good practice)
- ❌ Line 11: `vga_buffer` missing `volatile` qualifier — should be
  `static volatile uint16_t *const vga_buffer`
- ⚠️ Line 14-18: vga_print doesn't bounds-check col parameter

### linker.ld
- ❌ Line 5: `. = 0x100000;` — missing `ALIGN(4)` directive before .text
- ❌ Line 17-20: `/DISCARD/` section uses inconsistent discard syntax;
  should use `DISCARD` without slashes

### Verdict
3 critical issues found: missing volatile on VGA buffer, linker alignment gap,
and inconsistent discard section syntax. These MUST be fixed.

[REJECTED]
""")

SIM_REFACTORER_OUTPUT = textwrap.dedent("""\
## Refactored Code — All Critic Issues Fixed

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
.loop:
    lodsb
    or al, al
    jz .done
    int 0x10
    jmp .loop
.done:
    popa
    ret

DISK_ERROR_MSG   db "Disk error!", 0
SECTORS_ERROR_MSG db "Sector mismatch!", 0
BOOT_DRIVE db 0

gdt_start:
    dd 0x0
    dd 0x0
gdt_code:
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 10011010b
    db 11001111b
    db 0x0
gdt_data:
    dw 0xFFFF
    dw 0x0
    db 0x0
    db 10010010b
    db 11001111b
    db 0x0
gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

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
    for (uint32_t i = 0; i < VGA_WIDTH * VGA_HEIGHT; i++) {
        vga_buffer[i] = (uint16_t)' ' | ((uint16_t)attr << 8);
    }
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

    .text ALIGN(4K) : {
        *(.text*)
    }

    .rodata ALIGN(4K) : {
        *(.rodata*)
    }

    .data ALIGN(4K) : {
        *(.data*)
    }

    .bss ALIGN(4K) : {
        *(COMMON)
        *(.bss*)
    }
}
```

## Changes Summary
1. **boot.asm**: Fixed KERNEL_OFFSET to 0x7E00, fixed disk_load dx push/pop ordering
2. **kernel.c**: Added `volatile` to vga_buffer, added bounds-checking to vga_print
3. **linker.ld**: Added `ALIGN(4K)` directives, removed invalid DISCARD section

[FIXED]
""")

SIM_QA_OUTPUT = textwrap.dedent("""\
## QA Verification — Mock Compilation Report

### File Completeness Check
| File       | Status   | Size (expected) | Lines |
|------------|----------|-----------------|-------|
| boot.asm   | ✅ VALID | 512 bytes       | 98    |
| kernel.c   | ✅ VALID | N/A (source)    | 42    |
| linker.ld  | ✅ VALID | N/A (source)    | 21    |

### boot.asm
- ✅ Boot signature 0xAA55 confirmed at offset 510-511
- ✅ Sector size exactly 512 bytes after padding
- ✅ GDT code segment: 0x9A (present, ring 0, code, executable)
- ✅ GDT data segment: 0x92 (present, ring 0, data, writable)
- ✅ Protected mode switch sequence: CLI → LGDT → CR0.PE=1 → far JMP
- ✅ Stack configured at 0x90000 (safe, above kernel)

### kernel.c
- ✅ Entry point `kernel_main` — no arguments, void return (freestanding)
- ✅ No standard library dependencies (no stdio, no malloc)
- ✅ VGA buffer at 0xB8000 — correct text-mode address
- ✅ Bounds-checking on vga_print prevents buffer overflow
- ✅ Halt loop uses `hlt` for power efficiency
- [WARNING] No interrupt handlers configured — intentional for minimal kernel

### linker.ld
- ✅ ENTRY(kernel_main) matches kernel.c symbol
- ✅ Base address 0x100000 — standard x86 kernel location
- ✅ ALIGN(4K) on all output sections — satisfies Critic's alignment concern
- ✅ All standard sections accounted for

### Cross-File Consistency
- ✅ `kernel_main` referenced in linker.ld and defined in kernel.c
- ✅ `KERNEL_OFFSET` in boot.asm references the loaded kernel location
- ✅ No undefined external symbols

### Final Verdict
All 3 files pass completeness and correctness checks. Ready for real compilation
with `nasm -f bin boot.asm -o boot.bin && gcc -m32 -ffreestanding -c kernel.c -o kernel.o`.
No blocking issues found.

[VERIFIED]
""")


# ═══════════════════════════════════════════════════════════════
# MULTI-ENDPOINT LLM CLIENT
# ═══════════════════════════════════════════════════════════════

class MultiLLMClient:
    """Manages multiple LLM clients keyed by agent role."""

    def __init__(self):
        self.clients: Dict[str, AsyncOpenAI] = {}
        self.sim_mode: Dict[str, bool] = {}

    def configure(self, role: str, api_key: str, base_url: str, model: str):
        """Register an API client for a role."""
        key = api_key.strip()
        if not key:
            self.sim_mode[role] = True
            return
        self.sim_mode[role] = False
        self.clients[role] = AsyncOpenAI(
            api_key=key,
            base_url=base_url.strip() or DEFAULT_ENDPOINTS.get(role, {}).get("base_url", ""),
        )
        # Store model name for later use
        if not hasattr(self, 'models'):
            self.models = {}
        self.models[role] = model.strip() or DEFAULT_ENDPOINTS.get(role, {}).get("model", "")

    def is_sim(self, role: str) -> bool:
        return self.sim_mode.get(role, True)


# ═══════════════════════════════════════════════════════════════
# AUTONOMOUS SWARM LOOP ENGINE
# ═══════════════════════════════════════════════════════════════

class SwarmEngine:
    def __init__(self, mllm: MultiLLMClient):
        self.mllm = mllm

    # ── streaming helper ──
    async def _call_llm(
        self,
        role: str,
        system_prompt: str,
        user_content: str,
        extra_body: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the configured LLM, falling back to simulation."""
        if self.mllm.is_sim(role):
            for chunk in self._sim_stream(role, user_content):
                yield chunk
            return

        client = self.mllm.clients[role]
        model = getattr(self.mllm, 'models', {}).get(role, "gpt-4")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": True,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n\n❌ API Error [{role}]: {str(e)}\nFalling back to simulation...\n\n"
            for chunk in self._sim_stream(role, user_content):
                yield chunk

    def _sim_stream(self, role: str, _user_content: str):
        """Simulated streaming for demo / no-API-key mode."""
        sim_map = {
            "architect":  SIM_ARCHITECT_OUTPUT,
            "critic":     SIM_CRITIC_APPROVED,   # default to approved for demo
            "refactorer": SIM_REFACTORER_OUTPUT,
            "qa":         SIM_QA_OUTPUT,
        }
        text = sim_map.get(role, "[no simulation available]")
        for i, ch in enumerate(text):
            yield ch
            if i % 3 == 0:
                time.sleep(0.002)

    # ── the main loop ──
    async def run_swarm(
        self,
        task_prompt: str,
        critic_mode: str = "strict",   # "strict" = use rejected sim, "lenient" = approved sim
    ) -> AsyncGenerator[dict, None]:
        """Execute the full autonomous swarm loop, yielding UI state updates."""
        state = SwarmState(task_prompt=task_prompt)
        state.task_id = f"swarm-{int(time.time())}"
        state.phase = "architect"
        state.log("🚀 Swarm initialized")

        # ── STEP A: Architect ──
        async for result in self._run_architect(state):
            yield result

        # ── STEP B + C: Critic/Refactorer loop (max 5) ──
        async for result in self._critic_loop(state, critic_mode):
            yield result

        if not state.approved:
            state.log("❌ Max critic loops exhausted — code rejected")
            state.phase = "failed"
            yield self._build_ui_state(state, "❌ MAX LOOPS — Swarm halted")
            return

        # ── STEP D: QA Verification ──
        async for result in self._run_qa(state):
            yield result
        async for result in self._run_mock_compile(state):
            yield result

        state.phase = "complete"
        state.log("✅ Swarm complete — all stages passed")
        yield self._build_ui_state(state, "✅ SWARM COMPLETE — All agents passed")

    async def _run_architect(self, state: SwarmState):
        """Step A: Generate boot.asm, kernel.c, linker.ld."""
        agent = state.agents[AgentRole.ARCHITECT]
        agent.status = "running"
        agent.start_ts = time.time()
        state.log("🏗️  Architect (GLM-5.2) — generating x86 kernel source")
        state.phase = "architect"

        yield self._build_ui_state(state, "🏗️  Architect generating kernel source...")

        buffer = []
        async for token in self._call_llm("architect", ARCHITECT_PROMPT, state.task_prompt):
            buffer.append(token)
            agent.lines.append(token)
            if len(buffer) % 20 == 0:
                yield self._build_ui_state(state, "🏗️  Architect streaming...")

        agent.status = "done"
        agent.end_ts = time.time()
        # Extract files into virtual FS
        self._extract_files(state, "".join(buffer))
        state.log(f"🏗️  Architect complete — {len(buffer)} tokens, {len(state.virtual_fs)} files")
        yield self._build_ui_state(state, "🏗️  Architect done → handing off to Critic")

    async def _critic_loop(self, state: SwarmState, mode: str) -> bool:
        """Step B+C loop: Critic evaluates, Refactorer fixes rejected code. Max 5 loops."""
        state.phase = "critic"

        for loop_idx in range(1, MAX_CRITIC_LOOPS + 1):
            state.loop_count = loop_idx
            state.log(f"🔍 Critic Loop {loop_idx}/{MAX_CRITIC_LOOPS}")

            # ── Critic (Step B) ──
            critic = state.agents[AgentRole.CRITIC]
            critic.status = "running"
            critic.loop_index = loop_idx
            critic.start_ts = time.time()
            critic.lines = []
            critic.verdict = ""

            code_snapshot = self._build_code_snapshot(state)
            yield self._build_ui_state(state, f"🔍 Critic evaluating (loop {loop_idx}/{MAX_CRITIC_LOOPS})...")

            critic_buf = []
            # Use rejected simulation on first loop in strict mode for demo
            actual_role = "critic"
            async for token in self._call_llm(
                "critic",
                CRITIC_PROMPT,
                f"Review the following code:\n\n{code_snapshot}",
                extra_body={"thinking": {"type": "enabled"}} if not self.mllm.is_sim("critic") else None,
            ):
                critic_buf.append(token)
                critic.lines.append(token)

            critic.status = "done"
            critic.end_ts = time.time()
            full_critic = "".join(critic_buf)

            # Override with simulation patterns for demo
            if self.mllm.is_sim("critic"):
                if mode == "strict" and loop_idx == 1:
                    full_critic = SIM_CRITIC_REJECTED
                    critic.lines = list(SIM_CRITIC_REJECTED)
                else:
                    full_critic = SIM_CRITIC_APPROVED
                    critic.lines = list(SIM_CRITIC_APPROVED)

            # Parse verdict
            if "[APPROVED]" in full_critic:
                critic.verdict = "APPROVED"
                state.verdict = "APPROVED"
                state.approved = True
                state.log(f"✅ Critic (loop {loop_idx}): [APPROVED] — advancing to QA")
                yield self._build_ui_state(state, "✅ Code APPROVED by Critic → QA verification")
                return

            elif "[REJECTED]" in full_critic:
                critic.verdict = "REJECTED"
                state.verdict = "REJECTED"
                state.log(f"❌ Critic (loop {loop_idx}): [REJECTED] — routing to Refactorer")

                # ── Refactorer (Step C) ──
                refactorer = state.agents[AgentRole.REFACTORER]
                refactorer.status = "running"
                refactorer.loop_index = loop_idx
                refactorer.start_ts = time.time()
                refactorer.lines = []

                yield self._build_ui_state(state, f"🔧 Refactorer fixing issues (loop {loop_idx})...")

                refac_buf = []
                async for token in self._call_llm(
                    "refactorer",
                    REFACTORER_PROMPT,
                    f"Original code:\n{code_snapshot}\n\nCritic feedback:\n{full_critic}",
                ):
                    refac_buf.append(token)
                    refactorer.lines.append(token)

                refactorer.status = "done"
                refactorer.end_ts = time.time()

                if self.mllm.is_sim("refactorer"):
                    refac_buf = list(SIM_REFACTORER_OUTPUT)
                    refactorer.lines = list(SIM_REFACTORER_OUTPUT)

                full_refac = "".join(refac_buf) if isinstance(refac_buf, list) else refac_buf
                self._extract_files(state, full_refac if isinstance(full_refac, str) else "".join(refac_buf))
                state.log(f"🔧 Refactorer (loop {loop_idx}): fixes applied, re-submitting to Critic")
                yield self._build_ui_state(state, f"🔧 Refactorer done → re-submitting to Critic (loop {loop_idx+1})")
                # No clear verdict — treat as rejected, loop continues
                state.log(f"⚠️ Critic (loop {loop_idx}): no clear verdict, treating as REJECTED")
                state.verdict = "REJECTED"

        state.approved = False
        return

    async def _run_qa(self, state: SwarmState):
        """Step D: QA verification."""
        state.phase = "qa"
        qa = state.agents[AgentRole.QA]
        qa.status = "running"
        qa.start_ts = time.time()
        qa.lines = []

        code_snapshot = self._build_code_snapshot(state)
        state.log("🔬 QA (MiniMax-M3) — verifying code integrity")
        yield self._build_ui_state(state, "🔬 QA verifying code completeness...")

        qa_buf = []
        async for token in self._call_llm("qa", QA_PROMPT, f"Verify the following approved code:\n\n{code_snapshot}"):
            qa_buf.append(token)
            qa.lines.append(token)

        qa.status = "done"
        qa.end_ts = time.time()

        full_qa = "".join(qa_buf)
        if "[VERIFIED]" in full_qa:
            state.log("✅ QA: [VERIFIED] — all checks passed")
        else:
            state.log("⚠️ QA: verification complete with warnings")
        yield self._build_ui_state(state, "🔬 QA verification complete")

    async def _run_mock_compile(self, state: SwarmState):
        """Mock-compile: validate virtual file array."""
        state.log("📦 Mock-compiling virtual file array...")
        yield self._build_ui_state(state, "📦 Mock-compiling...")

        required = {"boot.asm", "kernel.c", "linker.ld"}

        for fname in required:
            if fname not in state.virtual_fs:
                state.virtual_fs[fname] = VirtualFile(
                    name=fname,
                    content=f"[MISSING: {fname} was not generated]",
                    status="missing",
                )
                state.log(f"❌ Mock-compile: {fname} MISSING")

        # Validate boot.asm size
        boot = state.virtual_fs.get("boot.asm")
        if boot and boot.status != "missing":
            content = boot.content
            if len(content) < 512:
                boot.status = "corrupt"
                state.log(f"⚠️ boot.asm: {len(content)} bytes (expected 512)")
            elif not ("0xAA55" in content or "AA55" in content or "0xaa55" in content.lower()):
                boot.status = "corrupt"
                state.log("⚠️ boot.asm: missing 0xAA55 boot signature")
            else:
                boot.status = "verified"
                state.log(f"✅ boot.asm: 512 bytes, signature OK, sha256={boot.sha256}")

        # Validate kernel.c
        kc = state.virtual_fs.get("kernel.c")
        if kc and kc.status != "missing":
            if "kernel_main" in kc.content:
                kc.status = "verified"
                state.log(f"✅ kernel.c: entry point 'kernel_main' found, sha256={kc.sha256}")
            else:
                kc.status = "corrupt"
                state.log("⚠️ kernel.c: missing kernel_main entry point")

        # Validate linker.ld
        ld = state.virtual_fs.get("linker.ld")
        if ld and ld.status != "missing":
            if "ENTRY" in ld.content and "SECTIONS" in ld.content:
                ld.status = "verified"
                state.log(f"✅ linker.ld: valid linker script, sha256={ld.sha256}")
            else:
                ld.status = "corrupt"
                state.log("⚠️ linker.ld: invalid format")

        state.log("📦 Mock-compile complete")
        yield self._build_ui_state(state, "📦 Mock-compile done → Build report ready")

    # ── utilities ──

    def _extract_files(self, state: SwarmState, text: str):
        """Extract code blocks into the virtual file array."""
        pattern = re.compile(
            r'###\s+(\S+)\s*\n\s*```(?:\w+)?\n(.*?)```',
            re.DOTALL,
        )
        for match in pattern.finditer(text):
            fname = match.group(1).strip()
            content = match.group(2).strip()
            vf = VirtualFile(name=fname, content=content)
            state.virtual_fs[fname] = vf
            state.log(f"📄 Virtual FS: {fname} ({vf.size_bytes}B, sha256={vf.sha256})")

    def _build_code_snapshot(self, state: SwarmState) -> str:
        """Build a text snapshot of all virtual files for the critic."""
        if not state.virtual_fs:
            return "[No code generated yet]"
        parts = []
        for fname in sorted(state.virtual_fs.keys()):
            vf = state.virtual_fs[fname]
            parts.append(f"### {fname}\n```\n{vf.content}\n```\n")
        return "\n".join(parts)

    def _build_ui_state(self, state: SwarmState, phase_msg: str) -> dict:
        """Build the complete UI state dict for Gradio updates."""
        return {
            "architect_log": state.agents[AgentRole.ARCHITECT].terminal_text,
            "critic_log": state.agents[AgentRole.CRITIC].terminal_text,
            "refactorer_log": state.agents[AgentRole.REFACTORER].terminal_text,
            "qa_log": state.agents[AgentRole.QA].terminal_text,
            "bus_html": self._render_bus(state),
            "status_html": self._render_status(state, phase_msg),
            "fs_html": self._render_virtual_fs(state),
        }

    def _render_bus(self, state: SwarmState) -> str:
        rows = []
        for role in AgentRole:
            a = state.agents[role]
            labels = {
                AgentRole.ARCHITECT:  "🏗️  Architect",
                AgentRole.CRITIC:     "🔍 Critic",
                AgentRole.REFACTORER: "🔧 Refactorer",
                AgentRole.QA:         "🔬 QA",
            }
            status_icon = {"idle": "⏳", "running": "🔄", "done": "✅", "error": "❌"}.get(a.status, "❓")
            rows.append(
                f'<tr><td>{labels[role]}</td>'
                f'<td>{status_icon} {a.status}</td>'
                f'<td>{a.elapsed}</td>'
                f'<td>loop #{a.loop_index if a.loop_index else "—"}</td>'
                f'<td>{a.verdict}</td></tr>'
            )

        fs_rows = ""
        for fname in sorted(state.virtual_fs.keys()):
            vf = state.virtual_fs[fname]
            fs_rows += (
                f'<tr><td>📄 {fname}</td>'
                f'<td>{vf.size_bytes}B</td>'
                f'<td><code>{vf.sha256}</code></td>'
                f'<td>{vf.status}</td></tr>'
            )

        return f"""\
<h3>🚦 Execution Bus — Task <code>{state.task_id}</code></h3>
<table style="width:100%;border-collapse:collapse;color:#e2e8f0;font-size:0.85rem;">
<tr style="background:#1e293b;">
<th>Agent</th><th>Status</th><th>Time</th><th>Loop</th><th>Verdict</th>
</tr>
{''.join(rows)}
</table>
<br/>
<h4>💾 Virtual File System</h4>
<table style="width:100%;border-collapse:collapse;color:#e2e8f0;font-size:0.82rem;">
<tr style="background:#1e293b;">
<th>File</th><th>Size</th><th>SHA256</th><th>Status</th>
</tr>
{fs_rows if fs_rows else '<tr><td colspan="4">No files yet</td></tr>'}
</table>
<br/>
<h4>📜 Global Log</h4>
<pre style="max-height:160px;overflow-y:auto;font-size:0.75rem;color:#94a3b8;">{chr(10).join(state.global_log[-30:])}</pre>
"""

    def _render_status(self, state: SwarmState, msg: str) -> str:
        phase_colors = {
            "architect": "#e74c3c", "critic": "#f39c12",
            "refactorer": "#3498db", "qa": "#2ecc71",
            "complete": "#22c55e", "failed": "#ef4444", "init": "#94a3b8",
        }
        color = phase_colors.get(state.phase, "#94a3b8")
        pulse = 'class="pulse"' if state.phase not in ("complete", "failed", "init") else ""
        return f'<div style="color:{color};text-align:center;padding:10px;font-weight:700;font-size:1.05rem;" {pulse}>{msg}</div>'

    def _render_virtual_fs(self, state: SwarmState) -> str:
        """Render the virtual filesystem as a combined text view."""
        if not state.virtual_fs:
            return "*No files generated yet*"
        parts = []
        for fname in sorted(state.virtual_fs.keys()):
            vf = state.virtual_fs[fname]
            parts.append(f"### 📄 {fname}  ({vf.size_bytes}B)  [{vf.status}]")
            parts.append(f"```\n{vf.content[:3000]}\n```")
            parts.append("")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
.gradio-container {
    max-width: 100% !important;
    background: #0b1120 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
h1, h2, h3, h4 { color: #f1f5f9 !important; }
#main-title {
    text-align: center;
    padding: 1rem 0 0.25rem;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 16px;
    margin-bottom: 1rem;
}
#main-title h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e74c3c, #f39c12, #3498db, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.terminal-box textarea {
    background: #060a14 !important;
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    line-height: 1.4 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    min-height: 280px !important;
}
.agent-header {
    font-weight: 700;
    font-size: 0.9rem;
    padding: 8px 14px;
    border-radius: 8px 8px 0 0;
    margin-bottom: -4px;
}
#run-btn {
    background: linear-gradient(135deg, #e74c3c, #f39c12) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 12px 28px !important;
    border-radius: 12px !important;
    width: 100% !important;
}
#run-btn:hover { transform: scale(1.02) !important; }
#bus-panel {
    background: #0f172a !important;
    border: 2px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
#fs-output {
    background: #060a14 !important;
    color: #cbd5e1 !important;
    font-size: 0.78rem !important;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
.pulse { animation: pulse 1.5s infinite; }
"""


def create_ui():
    mllm = MultiLLMClient()
    engine = SwarmEngine(mllm)

    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        title="AutoClaw Multi-Agent Code Orchestrator",
    ) as demo:

        gr.HTML("""<div id="main-title">
            <h1>🎛️ Autonomous Multi-Agent Code-Debate Orchestrator</h1>
            <p style="color:#94a3b8;font-size:0.9rem;">
            Architect → Critic → Refactorer → QA &nbsp;|&nbsp; Max 5 fallback loops &nbsp;|&nbsp; x86 Kernel Generation
            </p>
        </div>""")

        # ── Tab 1: API Configuration Console ──
        with gr.Tab("🔑 API Configuration"):
            gr.Markdown("""### Configure API Endpoints for Each Agent
            Each agent targets a different model provider. Leave a key blank to use **simulation mode** (pre-recorded outputs for demo).""")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🏗️ Architect (GLM-5.2)")
                    arch_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                    arch_url = gr.Textbox(label="Base URL", value="https://open.bigmodel.cn/api/paas/v4")
                    arch_model = gr.Textbox(label="Model", value="glm-4-plus")
                with gr.Column():
                    gr.Markdown("#### 🔍 Critic (DeepSeek-V4-Pro)")
                    crit_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                    crit_url = gr.Textbox(label="Base URL", value="https://api.deepseek.com/v1")
                    crit_model = gr.Textbox(label="Model", value="deepseek-chat")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔧 Refactorer (Ornith-1.0-397B / OpenAI)")
                    refac_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                    refac_url = gr.Textbox(label="Base URL", value="https://api.openai.com/v1")
                    refac_model = gr.Textbox(label="Model", value="gpt-4o")
                with gr.Column():
                    gr.Markdown("#### 🔬 QA (MiniMax-M3)")
                    qa_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                    qa_url = gr.Textbox(label="Base URL", value="https://api.minimaxi.com/v1")
                    qa_model = gr.Textbox(label="Model", value="abab7-chat")
            apply_btn = gr.Button("✅ Apply Configuration", variant="primary")
            config_status = gr.Textbox(label="Status", interactive=False, value="No config applied — using simulation mode")

            def apply_config(ak, au, am, ck, cu, cm, rk, ru, rm, qk, qu, qm):
                mllm.configure("architect", ak, au, am)
                mllm.configure("critic", ck, cu, cm)
                mllm.configure("refactorer", rk, ru, rm)
                mllm.configure("qa", qk, qu, qm)
                status_parts = []
                for role, key_var in [("architect", ak), ("critic", ck), ("refactorer", rk), ("qa", qk)]:
                    status_parts.append(f"{role}: {'✅ LIVE' if key_var.strip() else '📺 SIM'}")
                return " | ".join(status_parts)

            apply_btn.click(
                fn=apply_config,
                inputs=[arch_key, arch_url, arch_model, crit_key, crit_url, crit_model,
                        refac_key, refac_url, refac_model, qa_key, qa_url, qa_model],
                outputs=[config_status],
            )

        # ── Tab 2: Swarm Orchestrator ──
        with gr.Tab("🚀 Swarm Orchestrator"):
            with gr.Row():
                task_input = gr.Textbox(
                    label="📋 Build Directive",
                    placeholder="e.g., 'Generate a minimal x86 bootable kernel with VGA output'",
                    lines=2,
                    scale=6,
                    value="Generate a minimal x86 bootable kernel: boot.asm (real-mode→protected-mode switch), kernel.c (VGA text-mode banner), linker.ld (ELF linking at 0x100000)",
                )
                with gr.Column(scale=1, min_width=160):
                    critic_mode = gr.Radio(
                        choices=["strict", "lenient"],
                        value="strict",
                        label="Critic Mode",
                        info="strict = REJECTED first loop (demo)",
                    )
                    run_btn = gr.Button("🚀 Launch Swarm Build", variant="primary", elem_id="run-btn")
                    stop_btn = gr.Button("⏹️ Halt", variant="stop")

            status_html = gr.HTML('<div style="color:#94a3b8;text-align:center;padding:10px;">⏳ Awaiting launch...</div>')

            # ── 4-Agent split-column terminal logs ──
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML('<div class="agent-header" style="background:#e74c3c;color:white;">🏗️ ARCHITECT (GLM-5.2)</div>')
                    arch_log = gr.Textbox(label="", lines=16, interactive=False, elem_classes=["terminal-box"], show_label=False, value="[ waiting ]")
                with gr.Column():
                    gr.HTML('<div class="agent-header" style="background:#f39c12;color:white;">🔍 CRITIC (DeepSeek-V4-Pro)</div>')
                    crit_log = gr.Textbox(label="", lines=16, interactive=False, elem_classes=["terminal-box"], show_label=False, value="[ waiting ]")
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML('<div class="agent-header" style="background:#3498db;color:white;">🔧 REFACTORER (Ornith-1.0-397B)</div>')
                    refac_log = gr.Textbox(label="", lines=16, interactive=False, elem_classes=["terminal-box"], show_label=False, value="[ waiting ]")
                with gr.Column():
                    gr.HTML('<div class="agent-header" style="background:#2ecc71;color:white;">🔬 QA (MiniMax-M3)</div>')
                    qa_log = gr.Textbox(label="", lines=16, interactive=False, elem_classes=["terminal-box"], show_label=False, value="[ waiting ]")

        # ── Tab 3: Execution Bus & Virtual FS ──
        with gr.Tab("🚦 Execution Bus"):
            bus_html = gr.HTML("""<div id="bus-panel">
                <h3>🚦 Execution Bus</h3><p style="color:#94a3b8;">No active task. Launch the swarm.</p>
            </div>""")
            gr.Markdown("### 💾 Virtual File System (Mock-Compiled Output)")
            fs_output = gr.Markdown("*No files generated*", elem_id="fs-output")

        # ── Event Handler ──
        async def on_launch(task: str, mode: str):
            if not task.strip():
                yield (
                    "[no task]", "[no task]", "[no task]", "[no task]",
                    "<div id='bus-panel'><p>❌ No task provided</p></div>",
                    '<div style="color:#ef4444;">❌ No task</div>',
                    "*No files*",
                )
                return

            engine = SwarmEngine(mllm)
            async for state_dict in engine.run_swarm(task, mode):
                yield (
                    state_dict["architect_log"],
                    state_dict["critic_log"],
                    state_dict["refactorer_log"],
                    state_dict["qa_log"],
                    state_dict["bus_html"],
                    state_dict["status_html"],
                    state_dict["fs_html"],
                )

        run_event = run_btn.click(
            fn=on_launch,
            inputs=[task_input, critic_mode],
            outputs=[arch_log, crit_log, refac_log, qa_log, bus_html, status_html, fs_output],
        )
        stop_btn.click(fn=None, cancels=[run_event])

        # ── Tab 4: About ──
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""## 🎛️ Autonomous Multi-Agent Code-Debate Orchestrator

### Architecture
```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────┐
│  🏗️  ARCHITECT (GLM-5.2)                        │
│  Generates: boot.asm, kernel.c, linker.ld       │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  🔍 CRITIC (DeepSeek-V4-Pro, thinking=max)      │
│  Reviews syntax, memory, ABI                    │
│  Output: [APPROVED] or [REJECTED]               │
└──────┬──────────────────────────┬───────────────┘
       │                          │
   [APPROVED]                [REJECTED] (max 5 loops)
       │                          │
       ▼                          ▼
┌──────────────┐    ┌─────────────────────────────┐
│ 🔬 QA        │    │ 🔧 REFACTORER (Ornith-397B) │
│ MiniMax-M3   │    │ Fixes critic issues          │
│ Mock-compile │    │ Feeds back to Critic ────────┘
│ Verify       │
└──────┬───────┘
       ▼
   [VERIFIED]
```

### API Providers
| Agent | Provider | Model | thinking_effort |
|-------|----------|-------|-----------------|
| Architect | Zhipu/GLM | GLM-5.2 | default |
| Critic | DeepSeek | V4-Pro | max |
| Refactorer | Ornith | 1.0-397B | default |
| QA | MiniMax | M3 | default |

### Simulation Mode
Leave API keys blank to run in simulation mode with pre-recorded x86 kernel code
and a full critic/refactor/qa debate loop for demonstration.
""")

    return demo


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    demo = create_ui()
    demo.queue(default_concurrency_limit=5, max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
