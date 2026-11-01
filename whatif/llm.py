"""One interface to the language model, with a response cache, a call log, and a spending limit.

Every response is stored under the hash of model, settings, and prompt, so that a repeated call is answered from the
cache at no cost and every result can be traced to its prompt. Every call is logged with its token counts and cost.
Before a call, the spending since config.llm.budget_since plus the cost of the call at its token limit is compared with
config.llm.budget_usd, and the call is refused if it would exceed the limit. Models of OpenAI are called through the
Responses API, where responses are not stored by the provider, models of Anthropic through the Messages API with
adaptive thinking at the given effort, models of DeepSeek through their chat API, in thinking mode, and open models
under names such as ollama/olmo-3:7b, served locally by Ollama, or vllm/olmo-3.1-32b-think, served by vLLM at the
address and key in VLLM_URL and VLLM_KEY or in the file paths.yaml vllm_endpoint_file. Open models think before they
answer, so their JSON object is taken from the end of the answer, and their calls cost nothing.
"""
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from whatif.config import CONFIG, PATHS, ROOT

L = CONFIG["llm"]
CACHE = ROOT / "data" / "closed" / "llm_cache"
LOG = ROOT / "data" / "closed" / "llm_log.jsonl"
PROVIDERS = {"openai": (None, "OPENAI_API_KEY", "openai_key_file"),
             "anthropic": (None, "ANTHROPIC_API_KEY", "anthropic_key_file"),
             "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek_key_file"),
             "ollama": ("http://localhost:11434/v1", None, None),
             "vllm": (None, "VLLM_KEY", "vllm_endpoint_file")}
_clients = {}
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def provider(model):
    if model.startswith(("ollama/", "vllm/")):
        return model.split("/")[0]
    return "anthropic" if model.startswith("claude") else "deepseek" if model.startswith("deepseek") else "openai"


def client(name):
    if name not in _clients:
        from openai import OpenAI
        url, variable, file = PROVIDERS[name]
        key = os.environ.get(variable) if variable else "local"
        if name == "vllm":                                  # address and key of a server started in a notebook
            url = os.environ.get("VLLM_URL")
            if not (url and key) and PATHS.get(file):
                url, key = open(PATHS[file]).read().split()[:2]
            url = url.rstrip("/") + "/v1"
        elif not key and PATHS.get(file):
            found = re.search(r"sk-[A-Za-z0-9_\-]{20,}", open(PATHS[file], errors="ignore").read())
            key = found.group(0) if found else None
        if not key:
            raise RuntimeError(f"no key in {variable} or paths.yaml {file}")
        if name == "anthropic":
            from anthropic import Anthropic
            _clients[name] = Anthropic(api_key=key)
        else:
            _clients[name] = OpenAI(api_key=key, base_url=url)
    return _clients[name]


def call(model, effort, system, user, max_out):
    """Text, status, and token counts (input, cached input, output) of one response."""
    if provider(model) == "openai":
        r = client("openai").responses.create(model=model, instructions=system, input=user,
                                              reasoning={"effort": effort}, max_output_tokens=max_out,
                                              text={"format": {"type": "json_object"}}, store=False)
        u = r.usage
        return (r.output_text, r.status, u.input_tokens, getattr(u.input_tokens_details, "cached_tokens", 0) or 0,
                u.output_tokens)
    if provider(model) == "anthropic":                      # adaptive thinking, with its depth set by the effort
        r = client("anthropic").messages.create(model=model, system=system, max_tokens=max_out,
                                                messages=[{"role": "user", "content": user}],
                                                extra_body={"output_config": {"effort": effort}})
        text = "".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        return (text[text.find("{"):text.rfind("}") + 1], r.stop_reason, u.input_tokens,
                u.cache_read_input_tokens or 0, u.output_tokens)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    local = provider(model) in ("ollama", "vllm")
    mode = {} if local else {"response_format": {"type": "json_object"}}   # a JSON mode cuts the thinking short
    r = client(provider(model)).chat.completions.create(model=model.split("/", 1)[1] if local else model,
                                                        messages=messages, max_tokens=max_out, **mode)
    u = r.usage
    text = r.choices[0].message.content or ""
    if local:                                               # the JSON object after the thinking
        text = text.split("</think>")[-1]
        text = text[text.find("{"):text.rfind("}") + 1]
    return (text, r.choices[0].finish_reason, u.prompt_tokens, getattr(u, "prompt_cache_hit_tokens", 0) or 0,
            u.completion_tokens)


def cost(model, tokens_in, tokens_cached, tokens_out):
    if provider(model) in ("ollama", "vllm"):
        return 0.0
    p = L["prices"][model]                                  # USD per 1M tokens: input, cached input, output
    return ((tokens_in - tokens_cached) * p[0] + tokens_cached * p[1] + tokens_out * p[2]) / 1e6


def spent():
    if not LOG.exists():
        return 0.0
    rows = [json.loads(l) for l in open(LOG)]
    return sum(r["usd"] for r in rows if r["time"] >= L["budget_since"])


_logged = None


def logged_keys():
    """Keys that already have a line in the log, so that a batch answer read again on resume is not billed twice."""
    global _logged
    if _logged is None:
        _logged = {json.loads(l)["key"] for l in open(LOG)} if LOG.exists() else set()
    return _logged


def log(purpose, model, effort, key, status, tokens_in, tokens_cached, tokens_out, share=1.0):
    """One line per billed response; share is the part of the list price paid, 0.5 in a batch."""
    logged_keys().add(key)
    with _lock, open(LOG, "a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "purpose": purpose, "model": model,
                            "effort": effort, "key": key, "status": status, "tokens_in": tokens_in,
                            "tokens_cached": tokens_cached, "tokens_out": tokens_out,
                            "usd": round(share * cost(model, tokens_in, tokens_cached, tokens_out), 5)}) + "\n")


def request(system, user, model, effort):
    """The user prompt as sent and the cache key of the call."""
    if "json" not in user.lower():                        # required by the JSON output format of the API
        user += "\n\nRespond with a JSON object."
    return user, hashlib.sha256(json.dumps([model, effort, system, user]).encode()).hexdigest()


def store(key, model, effort, system, user, answer):
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{key}.json").write_text(json.dumps({"model": model, "effort": effort, "system": system, "user": user,
                                                   "answer": answer}, ensure_ascii=False))


def batch(items, model=None, effort=None, max_out=None, wait=60):
    """Answers to many calls, stored in the cache under the keys that ask() uses, so that the calling code then runs
    from the cache. Models of Anthropic and OpenAI are called through their batch interfaces at half the price, other
    models through parallel single calls. items are (system, user, purpose) triples. Returns the numbers of answers
    stored and failed."""
    model, effort, max_out = model or L["model"], effort or L["effort"], max_out or L["max_output_tokens"]
    todo = {}
    for system, user, purpose in items:
        user, key = request(system, user, model, effort)
        if not (CACHE / f"{key}.json").exists():
            todo[key] = (system, user, purpose)
    if not todo:
        return 0, 0
    if provider(model) not in ("anthropic", "openai"):        # no batch interface, for example DeepSeek
        with ThreadPoolExecutor(L.get("parallel_calls", 16)) as ex:     # DeepSeek limits concurrency by balance
            done = list(ex.map(lambda v: _try(ask, v[0], v[1], v[2], model, effort, max_out), todo.values()))
        return sum(done), len(done) - sum(done)
    stored = failed = 0
    for key, text, status, tokens in _resume(todo, model, wait):  # batches submitted before this process started
        s, f = _keep(todo, key, text, status, tokens, model, effort)
        stored, failed = stored + s, failed + f
    todo = {k: v for k, v in todo.items() if not (CACHE / f"{k}.json").exists()}
    if not todo:
        return stored, failed
    worst = sum(cost(model, len(sy + us) // 3, 0, max_out) for sy, us, _ in todo.values()) / 2
    if spent() + worst > L["budget_usd"]:
        raise BudgetExceeded(f"spent {spent():.2f} USD, batch up to {worst:.2f} USD, limit {L['budget_usd']} USD")
    results = (_anthropic if provider(model) == "anthropic" else _openai)(todo, model, effort, max_out, wait)
    for key, text, status, tokens in results:
        s, f = _keep(todo, key, text, status, tokens, model, effort)
        stored, failed = stored + s, failed + f
    return stored, failed


def _keep(todo, key, text, status, tokens, model, effort):
    """Log and store one answer of a batch; returns the numbers stored and failed."""
    if key not in todo or (CACHE / f"{key}.json").exists():
        return 0, 0
    system, user, purpose = todo[key]
    if tokens and key not in logged_keys():                 # an answer read again on resume was logged already
        log(purpose, model, effort, key, status, *tokens, share=0.5)
    if not text:                                            # no output, usually all tokens spent on reasoning
        return 0, 1
    try:
        store(key, model, effort, system, user, json.loads(text[text.find("{"):text.rfind("}") + 1]))
        return 1, 0
    except (json.JSONDecodeError, TypeError, AttributeError):  # left to ask(), which retries it as a single call
        return 0, 1


def _try(f, *args):
    try:
        f(*args)
        return True
    except Exception:                                       # a failed call is left to ask() in the calling code
        return False


def _record(provider_name, batch_id, n, model=None, keys=()):
    """The identifier and keys of a submitted batch, so that it can be collected later if this process ends."""
    with open(LOG.parent / "batches.jsonl", "a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "provider": provider_name,
                            "id": batch_id, "n": n, "model": model, "keys": list(keys)}) + "\n")


def _resume(todo, model, wait):
    """Answers from batches of the same model, submitted in the past day, that hold keys still to be answered."""
    path = LOG.parent / "batches.jsonl"
    if not path.exists():
        return
    since = datetime.now().timestamp() - 86400
    for r in [json.loads(l) for l in open(path)]:
        if (r.get("model") != model or not set(r.get("keys", ())) & set(todo)
                or datetime.fromisoformat(r["time"]).timestamp() < since):
            continue
        if r["provider"] == "anthropic":
            c = client("anthropic").messages.batches
            while c.retrieve(r["id"]).processing_status != "ended":
                time.sleep(wait)
            yield from _anthropic_results(c, r["id"])
        else:
            try:
                yield from _openai_results(client("openai"), r["id"], wait)
            except RuntimeError:                            # a cancelled or failed batch without answers
                continue


def _chunks(todo):
    """Parts of a batch of at most about config llm.batch_chars characters each, since one large upload fails on a
    slow connection (2026-10-31: uploads above 2 MB broke off while Dropbox was syncing)."""
    limit = L.get("batch_chars", 600_000)
    part, size = {}, 0
    for k, v in todo.items():
        n = len(v[0]) + len(v[1])
        if part and size + n > limit:
            yield part
            part, size = {}, 0
        part[k] = v
        size += n
    if part:
        yield part


def _retry(f, tries=4):
    """A call to a provider, repeated after connection errors with growing pauses."""
    for a in range(tries):
        try:
            return f()
        except Exception as e:                              # connection errors of either client library
            if a == tries - 1 or not any(w in type(e).__name__ + str(e) for w in ("onnection", "imeout", "imed out")):
                raise
            time.sleep(30 * 2 ** a)


def _anthropic(todo, model, effort, max_out, wait):
    """Key, text, status, and token counts of each answer of a Message Batch, submitted in parts."""
    c = client("anthropic").messages.batches
    ids = []
    for part in _chunks(todo):
        b = _retry(lambda: c.create(requests=[{"custom_id": key, "params": {
            "model": model, "system": sy, "max_tokens": max_out, "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": us}]}} for key, (sy, us, _) in part.items()]))
        _record("anthropic", b.id, len(part), model, part)
        ids.append(b.id)
    for bid in ids:
        while c.retrieve(bid).processing_status != "ended":
            time.sleep(wait)
        yield from _anthropic_results(c, bid)


def _anthropic_results(c, batch_id):
    for r in c.results(batch_id):
        if r.result.type != "succeeded":
            yield r.custom_id, None, r.result.type, None
            continue
        m, u = r.result.message, r.result.message.usage
        yield (r.custom_id, "".join(x.text for x in m.content if x.type == "text"), m.stop_reason,
               (u.input_tokens, u.cache_read_input_tokens or 0, u.output_tokens))


def _openai(todo, model, effort, max_out, wait):
    """Key, text, status, and token counts of each answer of a batch of the Responses API, submitted in parts."""
    c = client("openai")
    ids = []
    for part in _chunks(todo):
        lines = [json.dumps({"custom_id": key, "method": "POST", "url": "/v1/responses", "body": {
            "model": model, "instructions": sy, "input": us, "reasoning": {"effort": effort},
            "max_output_tokens": max_out, "text": {"format": {"type": "json_object"}}, "store": False}})
                 for key, (sy, us, _) in part.items()]
        upload = _retry(lambda: c.files.create(file=("batch.jsonl", "\n".join(lines).encode()), purpose="batch"))
        b = _retry(lambda: c.batches.create(input_file_id=upload.id, endpoint="/v1/responses", completion_window="24h"))
        _record("openai", b.id, len(part), model, part)
        ids.append(b.id)
    for bid in ids:
        yield from _openai_results(c, bid, wait)


def _openai_results(c, batch_id, wait):
    while (b := c.batches.retrieve(batch_id)).status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(wait)
    if not b.output_file_id:
        raise RuntimeError(f"batch {b.id} {b.status}")
    for line in c.files.content(b.output_file_id).text.splitlines():
        r = json.loads(line)
        body = (r.get("response") or {}).get("body") or {}
        text = "".join(x.get("text", "") for o in body.get("output", []) if o.get("type") == "message"
                       for x in o.get("content", []) if x.get("type") == "output_text")
        u = body.get("usage") or {}
        tokens = (u.get("input_tokens", 0), (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                  u.get("output_tokens", 0)) if u else None
        yield r["custom_id"], text or None, body.get("status"), tokens


def ask(system, user, purpose, model=None, effort=None, max_out=None):
    """JSON answer of the model to a system and a user prompt, from the cache if the same call was made before."""
    model, effort, max_out = model or L["model"], effort or L["effort"], max_out or L["max_output_tokens"]
    user, key = request(system, user, model, effort)
    hit = CACHE / f"{key}.json"
    if hit.exists():
        return json.loads(hit.read_text())["answer"]

    worst = cost(model, len(system + user) // 3, 0, max_out)
    with _lock:
        if spent() + worst > L["budget_usd"]:
            raise BudgetExceeded(f"spent {spent():.2f} USD, limit {L['budget_usd']} USD")
    err = None
    for attempt in range(3):
        try:
            text, status, *tokens = call(model, effort, system, user, max_out)
        except Exception as e:                              # network and API errors are retried
            err = e
            time.sleep(2 ** attempt)
            continue
        log(purpose, model, effort, key, status, *tokens)   # every billed response counts toward the limit
        try:
            answer = json.loads(text)
            break
        except (json.JSONDecodeError, TypeError):           # incomplete answers, usually at the token limit
            err = RuntimeError(f"no JSON answer, status {status}")
    else:
        raise err if isinstance(err, RuntimeError) else RuntimeError(f"{type(err).__name__}: {str(err)[:160]}")
    store(key, model, effort, system, user, answer)
    return answer
