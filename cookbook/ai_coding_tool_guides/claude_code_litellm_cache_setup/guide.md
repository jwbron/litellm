# Claude Code → LiteLLM → Qwen/DeepSeek, with prompt caching (one-shot setup)

`cllm-setup` is a single, self-contained script that stands up a **host-side
LiteLLM proxy** and wires **Claude Code** to drive non-Anthropic models
(Qwen / DeepSeek via OpenRouter) — while real Claude traffic keeps going
straight to `api.anthropic.com`. It runs on **Fedora/Linux** (systemd user
unit) and **macOS** (launchd LaunchAgent).

It is built for **this fork**, which carries the `cache_control` passthrough
fixes that make prompt caching actually land on **Qwen** through the Anthropic
`/v1/messages` → OpenRouter path (without them, every turn pays the full input
rate and `cache_read_input_tokens` stays at 0). DeepSeek still works as a
routed model — its caching is automatic/prefix-based and ignores
`cache_control`, so it needs no patch. The relevant changes:

- `litellm/llms/openrouter/chat/transformation.py` — `QWEN` added to
  `CacheControlSupportedModels` (DeepSeek deliberately omitted: listing it
  would be inert).
- `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`
  — `is_anthropic_claude_model` widened to Qwen, and the
  `x-anthropic-billing-header:` system block dropped during Anthropic→OpenAI
  translation (its per-request hash otherwise busts the prefix cache).

## What it sets up

1. **`uv`** (user-level, no sudo) if missing.
2. **`litellm[proxy]` from this fork**, pinned to a commit, via `uv tool install`.
3. **`~/.config/litellm/litellm.env`** — holds `OPENROUTER_API_KEY`. No master
   key: the proxy binds to `127.0.0.1` and runs in LiteLLM's no-auth dev mode,
   so Claude Code's bearer is discarded and the real key is injected upstream.
4. **`~/.config/litellm/cost_callback.py`** — a custom logger that records
   authoritative per-session cost + cache stats (pulled from the upstream
   OpenRouter `usage`, which is otherwise stripped by the Anthropic
   translation).
5. **`~/.config/litellm/config.yaml`** — generated **interactively** (offers
   three recommended models, then lets you add your own with provider pinning
   and `reasoning_effort`). `CLLM_DEFAULTS=1` skips the prompts.
6. **A background service** — a shared run-wrapper driven by a systemd `--user`
   unit (Linux) or a launchd LaunchAgent (macOS).
7. **`cllm`** — a Claude Code launcher that points it at the proxy and sets the
   non-Claude-route env (1M-context suffix, subagent model pinning, capability
   gating, English-language anchor).
8. **Optional harness extras** — a plain statusline that splices in the real
   cost + cache-hit rate, a conditional DuckDuckGo MCP server, a hook that
   blocks the (non-functional on this route) built-in WebSearch/WebFetch, and
   the `settings.json` / `~/.claude.json` wiring for them.

## Prerequisites

- `jq`, `python3`, `curl` (the script prints the `dnf`/`brew` command if any
  are missing).
- An [OpenRouter](https://openrouter.ai) API key.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview).

## Usage

```bash
# Interactive (prompts for the OpenRouter key and the model list):
./cllm-setup

# Non-interactive (reuse an existing key, seed the 3 default models):
CLLM_DEFAULTS=1 ./cllm-setup </dev/null
```

Then start a routed session:

```bash
cllm --model qwen3.7-max      # routed through the proxy
claude                        # unchanged: direct to Anthropic via OAuth
```

## Notes

- Re-runnable. Existing `config.yaml`, env, and a customized statusline are
  detected and preserved (the default statusline self-identifies with a marker;
  anything lacking it is left alone). The `settings.json` / `~/.claude.json`
  patches are idempotent and refuse to overwrite an unparseable file.
- Loopback only. If you rebind off `127.0.0.1`, restore a `master_key` — the
  no-auth dev mode is only safe because the network surface is loopback.
