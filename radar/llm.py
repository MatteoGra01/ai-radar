"""Client LLM OpenAI-compatibile (OpenRouter di default) con scelta automatica dei modelli gratuiti,
fallback tra modelli/provider, scadenze reali per chiamata, budget di chiamate e di tempo, parsing JSON robusto.
Thread-safe: più chiamate possono girare in parallelo."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from datetime import datetime

from . import net
from .hardware import params_from_name


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]   llm {msg}", file=sys.stderr, flush=True)


def _parse_json(text: str):
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # cerca il primo oggetto/array JSON bilanciato
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
    return None


class Provider:
    def __init__(self, spec: dict, call_timeout: float):
        self.name = spec.get("name", "provider")
        self.base_url = spec["base_url"].rstrip("/")
        self.key = os.environ.get(spec.get("api_key_env", ""), "").strip()
        self.spec_models = spec.get("models", ["auto:free"])
        self.prefer = [p.lower() for p in spec.get("prefer", [])]
        self.call_timeout = call_timeout
        self.exhausted = False
        self._models: list[str] | None = None
        self._params: dict[str, set] = {}
        self._reasoning: dict[str, dict] = {}
        self._fails: dict[str, int] = {}
        self._lock = threading.Lock()
        self._next_slot = 0.0

    @property
    def available(self) -> bool:
        return bool(self.key) and not self.exhausted and bool(self.models())

    def _free_openrouter_models(self) -> list[str]:
        try:
            data = net.get_json(f"{self.base_url}/models", timeout=30)["data"]
        except Exception:
            return ["openrouter/free"]
        free = []
        for m in data:
            p = m.get("pricing") or {}
            out_mod = (m.get("architecture") or {}).get("output_modalities") or ["text"]
            is_free = m["id"].endswith(":free") or m["id"] == "openrouter/free"
            if not is_free or str(p.get("prompt")) not in ("0", "0.0") or "text" not in out_mod:
                continue
            if (m.get("context_length") or 0) < 32000:
                continue
            size = params_from_name(m["id"].split(":")[0])[0] or 30
            if size < 8:
                continue
            self._params[m["id"]] = set(m.get("supported_parameters") or [])
            self._reasoning[m["id"]] = m.get("reasoning") if isinstance(m.get("reasoning"), dict) else {}
            pref = next((i for i, p in enumerate(self.prefer) if p in m["id"].lower()), len(self.prefer))
            mandatory = bool((m.get("reasoning") or {}).get("mandatory")) if isinstance(m.get("reasoning"), dict) else False
            # prima i preferiti (misurati veloci e affidabili), poi il router, poi per dimensione;
            # in fondo quelli con ragionamento obbligatorio (lenti, spesso esauriscono i token senza rispondere)
            free.append((mandatory, pref, m["id"] != "openrouter/free", -min(size, 1000), m["id"]))
        free.sort()
        return [x[-1] for x in free] or ["openrouter/free"]

    def models(self) -> list[str]:
        with self._lock:
            if self._models is None:
                out: list[str] = []
                for m in self.spec_models:
                    if m == "auto:free":
                        out.extend(x for x in self._free_openrouter_models() if x not in out)
                    elif m not in out:
                        out.append(m)
                self._models = out
            return list(self._models)

    def demote(self, model: str, hard: bool = False):
        with self._lock:
            self._fails[model] = self._fails.get(model, 0) + 1
            if self._models and model in self._models:
                self._models.remove(model)
                if not hard and self._fails[model] < 2:
                    self._models.append(model)  # in fondo alla lista; alla seconda volta fuori per questa esecuzione

    def promote(self, model: str):
        with self._lock:
            if self._models and model in self._models and self._models[0] != model:
                self._models.remove(model)
                self._models.insert(0, model)

    def _wait_slot(self):
        # ≤ 20 richieste/minuto sui modelli free: una partenza ogni 3.2 s anche con thread paralleli
        with self._lock:
            now = time.time()
            start = max(now, self._next_slot)
            self._next_slot = start + 3.2
        if start > now:
            time.sleep(start - now)

    def _reasoning_off(self, model: str) -> dict:
        """Per una rassegna il 'ragionamento' serve a poco e consuma token/tempo: lo spegniamo o lo minimizziamo."""
        r = self._reasoning.get(model) or {}
        efforts = r.get("supported_efforts")
        if not r.get("mandatory") and (efforts is None or "none" in efforts):
            return {"effort": "none", "exclude": True}
        for eff in ("minimal", "low"):
            if efforts is None or eff in efforts:
                return {"effort": eff, "exclude": True}
        return {"max_tokens": 1024, "exclude": True}

    def chat(self, model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
        self._wait_slot()
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        params = self._params.get(model, set())
        if "response_format" in params:
            payload["response_format"] = {"type": "json_object"}
        if "reasoning" in params or model == "openrouter/free":
            payload["reasoning"] = self._reasoning_off(model)
        headers = {"Authorization": f"Bearer {self.key}", "X-Title": "AI Radar (personal)"}
        raw = net.fetch(f"{self.base_url}/chat/completions", data=payload, headers=headers,
                        timeout=60, retries=0, max_seconds=self.call_timeout)
        data = json.loads(raw)
        if "error" in data:
            err = data["error"]
            raise net.HttpError(str(err)[:300], err.get("code") if isinstance(err, dict) else None, str(err))
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        return msg.get("content") or ""


class LLM:
    def __init__(self, cfg: dict, enabled: bool = True):
        lc = cfg["llm"]
        self.max_calls = int(lc["max_calls_per_run"])
        self.temperature = float(lc["temperature"])
        self.deadline = time.time() + float(lc.get("time_budget_s", 1200))
        call_timeout = float(lc.get("call_timeout_s", 120))
        self.providers = [Provider(p, call_timeout) for p in lc.get("providers", [])] if enabled else []
        self.calls = 0
        self.failures = 0
        self.used_models: dict[str, int] = {}
        self.log: list[str] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def last_model(self) -> str | None:
        return getattr(self._local, "last_model", None)

    @property
    def available(self) -> bool:
        return (self.calls < self.max_calls and time.time() < self.deadline
                and any(p.available for p in self.providers))

    def _note(self, msg: str):
        with self._lock:
            self.log.append(msg)

    def chat_json(self, system: str, user: str, max_tokens: int = 4000, validate=None, label: str = ""):
        """Restituisce il JSON prodotto dal primo modello che risponde correttamente, oppure None."""
        for prov in self.providers:
            if not prov.available:
                continue
            for model in prov.models()[:5]:
                with self._lock:
                    if self.calls >= self.max_calls or time.time() > self.deadline:
                        self.log.append("budget LLM (chiamate o tempo) esaurito")
                        return None
                    self.calls += 1
                t0 = time.time()
                try:
                    text = prov.chat(model, system, user, max_tokens, self.temperature)
                except (net.HttpError, TimeoutError, OSError, ValueError) as e:
                    dt = round(time.time() - t0)
                    with self._lock:
                        self.failures += 1
                    status = getattr(e, "status", None)
                    body = (getattr(e, "body", "") or str(e)).lower()
                    self._note(f"{model}: {str(e)[:140]}")
                    _log(f"{label} ✗ {model} {dt}s: {str(e)[:100]}")
                    if status in (401, 402) or (status == 429 and ("per-day" in body or "per day" in body or "daily" in body)):
                        prov.exhausted = True  # chiave non valida, crediti o quota giornaliera finiti: prossimo provider
                        break
                    # 403 = modello non disponibile nella regione, 400/404 = modello non valido: fuori per questa esecuzione
                    prov.demote(model, hard=status in (400, 403, 404))
                    continue
                dt = round(time.time() - t0)
                parsed = _parse_json(text)
                if parsed is not None and (validate is None or validate(parsed)):
                    with self._lock:
                        key = f"{prov.name}/{model}"
                        self.used_models[key] = self.used_models.get(key, 0) + 1
                    self._local.last_model = model
                    prov.promote(model)
                    _log(f"{label} ✓ {model} {dt}s")
                    return parsed
                with self._lock:
                    self.failures += 1
                self._note(f"{model}: risposta non valida ({len(text)} caratteri)")
                _log(f"{label} ✗ {model} {dt}s: risposta non valida ({len(text)} caratteri)")
                prov.demote(model)
        return None
