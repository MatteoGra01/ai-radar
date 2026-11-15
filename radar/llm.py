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


TIERS = ("fast", "deep")  # fast = valutazione notizie ed editoriale, deep = estrazione schede/benchmark


def is_free(model: str) -> bool:
    return model.endswith(":free") or model == "openrouter/free"


class Provider:
    def __init__(self, spec: dict, call_timeout: float):
        self.name = spec.get("name", "provider")
        self.base_url = spec["base_url"].rstrip("/")
        self.key = os.environ.get(spec.get("api_key_env", ""), "").strip()
        self.spec = {"fast": spec.get("models", ["auto:free"]),
                     "deep": spec.get("deep_models") or spec.get("models", ["auto:free"])}
        self.prefer = [p.lower() for p in spec.get("prefer", [])]
        self.call_timeout = call_timeout
        self.exhausted = False
        self.paid_blocked = False  # tetto di spesa raggiunto: da qui in poi solo modelli gratuiti
        self._lists: dict[str, list[str]] = {}
        self._catalog: dict[str, dict] | None = None
        self._fails: dict[str, int] = {}
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _cat(self) -> dict[str, dict]:
        if self._catalog is None:
            try:
                data = net.get_json(f"{self.base_url}/models", timeout=30)["data"]
                self._catalog = {m["id"]: m for m in data}
            except Exception:
                self._catalog = {}
        return self._catalog

    def available(self, tier: str = "fast") -> bool:
        return bool(self.key) and not self.exhausted and bool(self.models(tier))

    def _free_models(self) -> list[str]:
        free = []
        for mid, m in self._cat().items():
            p = m.get("pricing") or {}
            out_mod = (m.get("architecture") or {}).get("output_modalities") or ["text"]
            if not is_free(mid) or str(p.get("prompt")) not in ("0", "0.0") or "text" not in out_mod:
                continue
            if (m.get("context_length") or 0) < 32000:
                continue
            size = params_from_name(mid.split(":")[0])[0] or 30
            if size < 8:
                continue
            pref = next((i for i, p in enumerate(self.prefer) if p in mid.lower()), len(self.prefer))
            r = m.get("reasoning") if isinstance(m.get("reasoning"), dict) else {}
            # prima i preferiti (misurati veloci e affidabili), poi il router, poi per dimensione;
            # in fondo quelli con ragionamento obbligatorio (lenti, spesso esauriscono i token senza rispondere)
            free.append((bool(r.get("mandatory")), pref, mid != "openrouter/free", -min(size, 1000), mid))
        free.sort()
        return [x[-1] for x in free] or ["openrouter/free"]

    def models(self, tier: str = "fast") -> list[str]:
        with self._lock:
            if tier not in self._lists:
                out: list[str] = []
                for m in self.spec[tier]:
                    if m == "auto:free":
                        out.extend(x for x in self._free_models() if x not in out)
                    elif m not in out:
                        out.append(m)
                self._lists[tier] = out
            lst = self._lists[tier]
            return [m for m in lst if is_free(m)] if self.paid_blocked else list(lst)

    def demote(self, model: str, hard: bool = False):
        with self._lock:
            self._fails[model] = self._fails.get(model, 0) + 1
            for lst in self._lists.values():
                if model in lst:
                    lst.remove(model)
                    if not hard and self._fails[model] < 2:
                        lst.append(model)  # in fondo; alla seconda volta fuori per questa esecuzione

    def _wait_slot(self, model: str):
        if not is_free(model):
            return
        # ≤ 20 richieste/minuto sui modelli free: una partenza ogni 3.2 s anche con thread paralleli
        with self._lock:
            now = time.time()
            start = max(now, self._next_slot)
            self._next_slot = start + 3.2
        if start > now:
            time.sleep(start - now)

    def _reasoning_off(self, model: str) -> dict:
        """Per una rassegna il 'ragionamento' serve a poco e consuma token/tempo: lo spegniamo o lo minimizziamo."""
        r = self._cat().get(model, {}).get("reasoning")
        r = r if isinstance(r, dict) else {}
        efforts = r.get("supported_efforts")
        if not r.get("mandatory") and (efforts is None or "none" in efforts):
            return {"effort": "none", "exclude": True}
        for eff in ("minimal", "low"):
            if efforts is None or eff in efforts:
                return {"effort": eff, "exclude": True}
        return {"max_tokens": 1024, "exclude": True}

    def chat(self, model: str, system: str, user: str, max_tokens: int, temperature: float) -> tuple[str, float]:
        """Restituisce (testo, costo in $)."""
        self._wait_slot(model)
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        params = set(self._cat().get(model, {}).get("supported_parameters") or [])
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
        cost = float((data.get("usage") or {}).get("cost") or 0)
        return msg.get("content") or "", cost


class LLM:
    def __init__(self, cfg: dict, enabled: bool = True):
        lc = cfg["llm"]
        self.max_calls = int(lc["max_calls_per_run"])
        self.max_cost = float(lc.get("max_cost_per_run_usd", 0.5))
        self.temperature = float(lc["temperature"])
        self.deadline = time.time() + float(lc.get("time_budget_s", 1200))
        call_timeout = float(lc.get("call_timeout_s", 120))
        self.providers = [Provider(p, call_timeout) for p in lc.get("providers", [])] if enabled else []
        self.calls = 0
        self.failures = 0
        self.cost = 0.0
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
                and any(p.available() for p in self.providers))

    def _note(self, msg: str):
        with self._lock:
            self.log.append(msg)

    def chat_json(self, system: str, user: str, max_tokens: int = 4000, validate=None, label: str = "",
                  tier: str = "fast"):
        """Restituisce il JSON prodotto dal primo modello che risponde correttamente, oppure None."""
        for prov in self.providers:
            if not prov.available(tier):
                continue
            for model in prov.models(tier)[:5]:
                if prov.paid_blocked and not is_free(model):
                    continue
                with self._lock:
                    if self.calls >= self.max_calls or time.time() > self.deadline:
                        self.log.append("budget LLM (chiamate o tempo) esaurito")
                        return None
                    self.calls += 1
                t0 = time.time()
                try:
                    text, cost = prov.chat(model, system, user, max_tokens, self.temperature)
                except (net.HttpError, TimeoutError, OSError, ValueError) as e:
                    dt = round(time.time() - t0)
                    with self._lock:
                        self.failures += 1
                    status = getattr(e, "status", None)
                    body = (getattr(e, "body", "") or str(e)).lower()
                    self._note(f"{model}: {str(e)[:140]}")
                    _log(f"{label} ✗ {model} {dt}s: {str(e)[:100]}")
                    if status == 401 or (status == 429 and ("per-day" in body or "per day" in body or "daily" in body)):
                        prov.exhausted = True  # chiave non valida o quota giornaliera finita: prossimo provider
                        break
                    if status == 402:  # crediti finiti: restano i gratuiti
                        prov.paid_blocked = True
                        continue
                    # 403 = modello non disponibile nella regione, 400/404 = modello non valido: fuori per questa esecuzione
                    prov.demote(model, hard=status in (400, 403, 404))
                    continue
                dt = round(time.time() - t0)
                with self._lock:
                    self.cost += cost
                    if self.cost >= self.max_cost and not prov.paid_blocked:
                        prov.paid_blocked = True
                        self.log.append(f"tetto di spesa {self.max_cost}$ raggiunto: solo modelli gratuiti")
                parsed = _parse_json(text)
                if parsed is not None and (validate is None or validate(parsed)):
                    with self._lock:
                        key = f"{prov.name}/{model}"
                        self.used_models[key] = self.used_models.get(key, 0) + 1
                    self._local.last_model = model
                    _log(f"{label} ✓ {model} {dt}s" + (f" ${cost:.4f}" if cost else ""))
                    return parsed
                with self._lock:
                    self.failures += 1
                self._note(f"{model}: risposta non valida ({len(text)} caratteri)")
                _log(f"{label} ✗ {model} {dt}s: risposta non valida ({len(text)} caratteri)")
                prov.demote(model)
        return None
