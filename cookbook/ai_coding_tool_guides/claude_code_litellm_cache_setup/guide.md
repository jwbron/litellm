# Claude Code → LiteLLM → any OpenRouter model, with working prompt caching (one-shot setup)

`cllm-setup` is a single, self-contained script that stands up a **host-side
LiteLLM proxy** and wires **Claude Code** to drive **any non-Anthropic model
available on OpenRouter** (Qwen, DeepSeek, GLM, Kimi, …) — while real Claude
traffic keeps going straight to `api.anthropic.com`. It runs on **Fedora/Linux**
(systemd user unit) and **macOS** (launchd LaunchAgent).

Routing itself is model-agnostic: add any OpenRouter model to the generated
`config.yaml` and `cllm --model <name>` will drive Claude Code at it. What this
**fork** adds is **prompt caching that actually lands** through the Anthropic
`/v1/messages` → OpenRouter path — without the fixes below, every turn pays the
full input rate and `cache_read_input_tokens` stays at 0.

## How caching works (and which models get it)

OpenRouter exposes two different caching mechanisms, and which one applies
depends on the upstream provider:

- **Explicit `cache_control` (breakpoint) caching** — the client must mark
  cacheable blocks with `cache_control: {"type": "ephemeral"}`. Required by
  Anthropic, Gemini, MiniMax, GLM / Z-AI, and **Qwen / Alibaba**.
- **Automatic prefix caching** — server-side, keyed on a stable request prefix;
  `cache_control` is ignored entirely. Used by **DeepSeek** and similar.

Both were broken through the Anthropic-translation path, and for both the
decisive fix is the same: Claude Code injects an `x-anthropic-billing-header`
system block whose per-request hash sits in the cached prefix and changes every
turn, so the prefix never matches and the cache never hits. On top of that,
Qwen also needs its `cache_control` markers to survive the Anthropic→OpenAI
translation, which stock LiteLLM did not preserve for it.

Models with no upstream caching at all still route fine; they just get no cache
discount.

### The relevant fork changes

- `litellm/llms/openrouter/chat/transformation.py` — `QWEN` added to
  `CacheControlSupportedModels`. LiteLLM already treated Claude / Gemini /
  MiniMax / GLM / Z-AI as `cache_control`-capable; this extends that to Qwen
  (Alibaba upstreams require explicit breakpoints). Without it the OpenRouter
  handler strips `cache_control` before the upstream call. **DeepSeek is
  deliberately omitted**: it auto-caches and ignores `cache_control`, so listing
  it would be inert.
- `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`
  — two changes in the Anthropic→OpenAI adapter:
  - `is_anthropic_claude_model` widened to recognize Qwen, so the adapter
    **attaches** `cache_control` to Qwen's translated blocks (it is gated on
    this check).
  - the `x-anthropic-billing-header:` system block is **dropped** during
    translation. This one is **model-agnostic**: its per-request hash otherwise
    busts the prefix cache for every provider, and it is what restores
    **DeepSeek's automatic caching** through this path as well as stabilizing
    Qwen's breakpoint.

Net effect: Qwen needs all three (attach + preserve + stable prefix); DeepSeek
needs only the billing-header drop; the other `cache_control` models already
worked upstream and benefit from the billing-header drop too.

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
5. **`~/.config/litellm/config.yaml`** — generated **interactively**. It seeds
   three recommended models (`qwen3.7-max`, `deepseek-v4-flash`,
   `deepseek-v4-pro`) as a starting point, then lets you add **any OpenRouter
   model** with provider pinning and `reasoning_effort`. `CLLM_DEFAULTS=1` skips
   the prompts.
6. **A background service** — a shared run-wrapper driven by a systemd `--user`
   unit (Linux) or a launchd LaunchAgent (macOS).
7. **`cllm`** (plus a `claude-litellm` alias) — a Claude Code launcher that
   points it at the proxy and sets the non-Claude-route env (1M-context suffix
   for true-1M models, subagent model pinning, capability gating,
   English-language anchor).
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

Then start a routed session at any model you configured:

```bash
cllm --model qwen3.7-max      # routed through the proxy
cllm --model deepseek-v4-pro  # any model in your config.yaml works
claude                        # unchanged: direct to Anthropic via OAuth
```

## Notes

- Re-runnable. Existing `config.yaml`, env, and a customized statusline are
  detected and preserved (the default statusline self-identifies with a marker;
  anything lacking it is left alone). The `settings.json` / `~/.claude.json`
  patches are idempotent and refuse to overwrite an unparseable file.
- Loopback only. If you rebind off `127.0.0.1`, restore a `master_key` — the
  no-auth dev mode is only safe because the network surface is loopback.
