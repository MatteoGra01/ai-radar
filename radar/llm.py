"""Client LLM OpenAI-compatibile (OpenRouter di default) con scelta automatica dei modelli gratuiti,
fallback tra modelli/provider, budget di chiamate e parsing JSON robusto."""
from __future__ import annotations

import json
import os
import re
import time

from . import net
from .hardware import params_from_name


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
    def __init__(self, spec: dict):
        self.name = spec.get("name", "provider")
        self.base_url = spec["base_url"].rstrip("/")
        self.key = os.environ.get(spec.get("api_key_env", ""), "").strip()
        self.spec_models = spec.get("models", ["auto:free"])
        self.exhausted = False
        self._models: list[str] | None = None
        self._supports_json: dict[str, bool] = {}
        self._last_call = 0.0

    @property
    def available(self) -> bool:
        return bool(self.key) and not self.exhausted

    def _free_openrouter_models(self) -> list[str]:
        try:
            data = net.get_json(f"{self.base_url}/models", timeout=30)["data"]
        except Exception:
            return ["openrouter/free"]
        free = []
        for m in data:
            p = m.get("pricing") or {}
            arch = m.get("architecture") or {}
            out_mod = arch.get("output_modalities") or ["text"]
            is_free = m["id"].endswith(":free") or m["id"] == "openrouter/free"
            if not is_free or str(p.get("prompt")) not in ("0", "0.0") or "text" not in out_mod:
                continue
            if (m.get("context_length") or 0) < 32000:
                continue
            self._supports_json[m["id"]] = "response_format" in (m.get("supported_parameters") or [])
            size = params_from_name(m["id"].split(":")[0])[0] or 30
            free.append((m["id"] != "openrouter/free", -min(size, 1000), m["id"]))
        free.sort()
        ids = [x[2] for x in free if (params_from_name(x[2].split(":")[0])[0] or 30) >= 8]
        return ids or ["openrouter/free"]

    def models(self) -> list[str]:
        if self._models is None:
            out: list[str] = []
            for m in self.spec_models:
                if m == "auto:free":
                    out.extend(x for x in self._free_openrouter_models() if x not in out)
                elif m not in out:
                    out.append(m)
            self._models = out
        return self._models

    def chat(self, model: str, system: str, user: str, max_tokens: int, temperature: float) -> str:
        wait = 3.2 - (time.time() - self._last_call)  # ≤ 20 richieste/minuto sui modelli free
        if wait > 0:
            time.sleep(wait)
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._supports_json.get(model):
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.key}", "X-Title": "AI Radar (personal)"}
        try:
            raw = net.fetch(f"{self.base_url}/chat/completions", data=payload, headers=headers,
                            timeout=240, retries=1)
        finally:
            self._last_call = time.time()
        data = json.loads(raw)
        if "error" in data:
            raise net.HttpError(str(data["error"])[:300], data["error"].get("code") if isinstance(data["error"], dict) else None)
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        return msg.get("content") or ""


class LLM:
    def __init__(self, cfg: dict, enabled: bool = True):
        lc = cfg["llm"]
        self.max_calls = int(lc["max_calls_per_run"])
        self.temperature = float(lc["temperature"])
        self.providers = [Provider(p) for p in lc.get("providers", [])] if enabled else []
        self.calls = 0
        self.failures = 0
        self.used_models: dict[str, int] = {}
        self.last_model: str | None = None
        self.log: list[str] = []

    @property
    def available(self) -> bool:
        return self.calls < self.max_calls and any(p.available for p in self.providers)

    def chat_json(self, system: str, user: str, max_tokens: int = 4000, validate=None):
        """Restituisce il JSON prodotto dal primo modello che risponde correttamente, oppure None."""
        for prov in self.providers:
            if not prov.available:
                continue
            for model in prov.models()[:6]:
                if self.calls >= self.max_calls:
                    self.log.append("budget di chiamate esaurito")
                    return None
                self.calls += 1
                try:
                    text = prov.chat(model, system, user, max_tokens, self.temperature)
                except net.HttpError as e:
                    self.failures += 1
                    body = (e.body or str(e)).lower()
                    self.log.append(f"{prov.name}/{model}: {str(e)[:160]}")
                    if e.status in (401, 402, 403) or (e.status == 429 and ("per-day" in body or "per day" in body or "daily" in body)):
                        prov.exhausted = True  # chiave non valida o quota giornaliera finita: prossimo provider
                        break
                    continue
                except Exception as e:
                    self.failures += 1
                    self.log.append(f"{prov.name}/{model}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                parsed = _parse_json(text)
                if parsed is not None and (validate is None or validate(parsed)):
                    key = f"{prov.name}/{model}"
                    self.used_models[key] = self.used_models.get(key, 0) + 1
                    self.last_model = model
                    # il modello che funziona va in cima per le chiamate successive
                    prov._models.remove(model)
                    prov._models.insert(0, model)
                    return parsed
                self.failures += 1
                self.log.append(f"{prov.name}/{model}: risposta non valida ({len(text)} caratteri)")
        return None
