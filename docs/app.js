"use strict";

/* ───────────── utilità ───────────── */
const GiB = 2 ** 30;
const QUANTS = [["BF16", 16], ["Q8_0", 8.5], ["Q6_K", 6.56], ["Q5_K_M", 5.69], ["Q4_K_M", 4.85], ["IQ4_XS", 4.25], ["Q3_K_M", 3.91], ["IQ2_M", 2.7]];
const KV_TYPES = [["F16", 2], ["Q8_0", 1.0625], ["Q4_0", 0.5625]];
const NATIVE = { FP8: 8.25, NVFP4: 4.5 };
const nativeDefs = () => ((S.index && S.index.hardware.native_formats) || []).filter((f) => NATIVE[f]).map((f) => [f, NATIVE[f]]).sort((a, b) => b[1] - a[1]);
const bpwOf = (q) => NATIVE[q] || (QUANTS.find(([n]) => n === q) || [])[1];
const CATS = { modello: "Modelli", feature: "Feature", tool: "Tool", paper: "Paper", guida: "Guide e analisi", industria: "Industria", altro: "Altro" };
const CAT1 = { modello: "Modello", feature: "Feature", tool: "Tool", paper: "Paper", guida: "Guida", industria: "Industria", altro: "Altro" };
const KIND = { open: "Pesi aperti", closed: "API", announced: "Annunciato", placeholder: "Senza pesi" };
const TEXT_PIPES = new Set(["text-generation", "image-text-to-text", "any-to-any", "text2text-generation", "conversational"]);
const OR_KEY = "radar-or-key";
const S = {
  index: null, models: {}, day: null, papers: null,
  cat: "tutte", minRel: 0, q: "", show: {},
  mFilter: "all", mSort: "recent", mQ: "", mLimit: 60,
  lbFilter: "all", lbSort: "intelligence", lbDir: -1, lbQ: "",
  pDays: 7, pSort: "score", pQ: "", pLimit: 40, topics: null,
  latestOnly: true, onlyFresh: false, fresh: new Set(), freshDone: {}, seenStart: null,
  emb: null, eBench: 0, eSort: "rank", eDir: 1, eQ: "", eOpen: false, eGpu: false, eLang: false, eSize: "all", eVram: "all",
  pView: "all", pOpen: new Set(), pDetails: null, archQ: "", archRes: null, archDays: {},
};
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "href") el.setAttribute("href", safeUrl(v));
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false || kid === "") continue;
    el.append(kid instanceof Node ? kid : String(kid));
  }
  return el;
}
function safeUrl(u) {
  if (typeof u !== "string") return "#";
  if (u.startsWith("#")) return u;
  try { const x = new URL(u); return ["http:", "https:"].includes(x.protocol) ? u : "#"; } catch { return "#"; }
}
const host = (u) => { try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return u; } };
function ext(href, ...kids) {
  const attrs = kids.length && kids[0] && typeof kids[0] === "object" && !(kids[0] instanceof Node) && !Array.isArray(kids[0]) ? kids.shift() : {};
  return h("a", { ...attrs, href, target: "_blank", rel: "noopener noreferrer" }, ...kids);
}
const MEM = {};
function store(k, v) {
  if (v === undefined) { try { const x = localStorage.getItem(k); if (x !== null) return x; } catch { /* storage non disponibile */ } return MEM[k] ?? null; }
  MEM[k] = v;
  try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch { /* resta in memoria */ }
  return null;
}
const isTextModel = (e) => !e.pipeline || TEXT_PIPES.has(e.pipeline);
const fmtTok = (t) => t >= 500 ? "500+" : `~${Math.round(t)}`;
const fmtDate = (d) => new Date(d + "T12:00:00").toLocaleDateString("it-IT", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
const fmtShort = (d) => d ? new Date(d.slice(0, 10) + "T12:00:00").toLocaleDateString("it-IT", { day: "numeric", month: "short", year: (new Date().getFullYear() + "" === d.slice(0, 4)) ? undefined : "numeric" }) : "";
function ago(iso) {
  if (!iso) return "";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  const rtf = new Intl.RelativeTimeFormat("it", { numeric: "auto" });
  if (s < 3600) return rtf.format(-Math.max(1, Math.round(s / 60)), "minute");
  if (s < 86400) return rtf.format(-Math.round(s / 3600), "hour");
  return rtf.format(-Math.round(s / 86400), "day");
}
function fmtCtx(n) {
  if (!n) return "";
  if (n >= 1e6 - 1) return (n % 1e6 === 0 ? n / 1e6 : n % 1048576 === 0 ? n / 1048576 : +(n / 1e6).toFixed(1)) + "M";
  if (n >= 1000) return (n % 1024 === 0 ? n / 1024 : Math.round(n / 1000)) + "k";
  return String(n);
}
function fmtParams(b) {
  if (!b) return "";
  if (b >= 1000) return +(b / 1000).toFixed(2) + "T";
  if (b >= 100) return Math.round(b) + "B";
  if (b >= 10) return +b.toFixed(1) + "B";
  return +b.toFixed(2) + "B";
}
const fmtPrice = (p) => p == null ? "?" : p === 0 ? "0" : p < 0.1 ? p.toFixed(3) : +p.toFixed(2);
const fmtNum = (x) => x == null ? "—" : +(+x).toFixed(1);
function isNew(e, ref) {
  const d = e.released || e.first_seen;
  if (!d) return false;
  const r = new Date((ref || new Date().toISOString()).slice(0, 10) + "T12:00:00");
  return (r - new Date(d.slice(0, 10) + "T12:00:00")) / 864e5 <= ((S.index && S.index.new_model_days) || 14);
}
function toast(msg) {
  const t = h("div", { class: "toast", role: "status" }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 3200);
}
async function getJSON(path) {
  const v = S.index ? S.index.updated : Date.now();
  const r = await fetch(`${path}?v=${encodeURIComponent(v)}`, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

/* ───────────── stima VRAM (stessa formula di radar/hardware.py) ───────────── */
function kvBytes(a, ctx, b = 2) {
  if (!a) return 0;
  let per;
  if (a.mla_dim) per = a.mla_dim * b;
  else if (a.kv_heads && a.head_dim) per = 2 * a.kv_heads * a.head_dim * b;
  else return 0;
  let x = (a.full_layers || 0) * per * ctx;
  if (a.sliding_layers) x += a.sliding_layers * per * Math.min(ctx, a.window || ctx);
  x += (a.linear_layers || 0) * (a.linear_state_bytes || 0);
  return x;
}
function vramRow(e, bpw, ctx, kvb = 2) {
  const hw = S.index.hardware;
  const w = e.params_b * 1e9 * bpw / 8 / GiB;
  const total = w + kvBytes(e.arch, ctx, kvb) / GiB + 0.75 + 0.02 * w;
  const act = e.active_b || e.params_b;
  const moe = e.active_b && e.active_b < e.params_b * 0.7;
  const eff = moe ? Math.min(0.7, 0.15 + 0.03 * act) : 0.7;
  const tok = eff * hw.bandwidth_gbs * 1e9 / (act * 1e9 * bpw / 8 + kvBytes(e.arch, 4096, kvb));
  return { w, total, fits: total <= hw.vram_gb - hw.reserved_gb, tok };
}
const kvKnown = (e) => !!(e.arch && (e.arch.mla_dim || (e.arch.kv_heads && e.arch.head_dim)));
// GPU + RAM: in VRAM KV cache, overhead e quanti più pesi possibile; il resto in RAM (velocità limitata dalla RAM)
function offloadRow(e, bpw, ctx, kvb = 2) {
  const hw = S.index.hardware;
  const ramAvail = Math.max(0, (hw.system_ram_gb || 0) - (hw.ram_reserved_gb ?? 8));
  const ramBw = hw.ram_bandwidth_gbs || 80;
  const w = e.params_b * 1e9 * bpw / 8 / GiB;
  const kv = kvBytes(e.arch, ctx, kvb) / GiB;
  const vramW = Math.max(0, hw.vram_gb - hw.reserved_gb - (hw.margin_gb ?? 1.5) - kv - 0.75 - 0.02 * w);
  const ramW = w - vramW;
  if (ramW <= 0 || ramW > ramAvail) return null;
  const frac = ramW / w;
  const act = e.active_b || e.params_b;
  const moe = e.active_b && e.active_b < e.params_b * 0.7;
  const eff = moe ? Math.min(0.7, 0.15 + 0.03 * act) : 0.7;
  const bytes = act * 1e9 * bpw / 8;
  const t = bytes * (1 - frac) / (eff * hw.bandwidth_gbs * 1e9) + bytes * frac / (0.6 * ramBw * 1e9);
  return { ram: ramW, vram: vramW + kv + 0.75 + 0.02 * w, tok: 1 / t };
}
function maxCtx(e, bpw, kvb = 2) {
  if (!e.params_b || !kvKnown(e)) return null;
  const hw = S.index.hardware;
  const usable = hw.vram_gb - hw.reserved_gb;
  const tot = (c) => vramRow(e, bpw, c, kvb).total;
  if (tot(512) > usable) return 0;
  let lo = 512, hi = e.arch.max_ctx || e.context || 1048576;
  if (tot(hi) <= usable) return hi;
  while (hi - lo > 256) { const mid = (lo + hi) >> 1; if (tot(mid) <= usable) lo = mid; else hi = mid; }
  return lo;
}
function vramTip(v) {
  if (!v.options) return v.text;
  return v.text + "\n\n" + v.options.map((o) => `${o.q}${o.native ? " (nativo)" : ""}: ${o.comfy ? "entra" : o.fits ? "al limite" : "non entra"}` +
    (o.fits && o.max_ctx ? `, contesto fino a ${fmtCtx(o.max_ctx)}` : "") + (o.fits && o.tok_s ? `, ~${o.tok_s >= 500 ? "500+" : o.tok_s} tok/s` : "")).join("\n");
}
const localModel = (e) => e.params_b && e.kind !== "closed" && e.kind !== "placeholder";

/* ───────────── confronto: selezione ───────────── */
function cmpGet() { try { return JSON.parse(store("radar-compare") || "[]").filter((id) => S.models[id]); } catch { return []; } }
function cmpSet(list) { store("radar-compare", JSON.stringify(list.slice(0, 4))); updateCmpUI(); }
function cmpToggle(id) {
  const l = cmpGet();
  const i = l.indexOf(id);
  if (i >= 0) l.splice(i, 1);
  else if (l.length >= 4) { toast("Puoi confrontare al massimo 4 modelli"); return; }
  else l.push(id);
  cmpSet(l);
}
function cmpBtn(e) {
  const on = cmpGet().includes(e.id);
  return h("button", { class: "cmp-btn" + (on ? " on" : ""), type: "button", "data-cmp": e.id, title: "Aggiungi o togli dal confronto",
    onclick: (ev) => { ev.stopPropagation(); cmpToggle(e.id); } }, on ? "✓ nel confronto" : "+ confronta");
}
function updateCmpUI() {
  const l = cmpGet();
  const badge = document.getElementById("cmp-count");
  badge.hidden = !l.length;
  badge.textContent = l.length;
  for (const b of document.querySelectorAll(".cmp-btn")) {
    const on = l.includes(b.dataset.cmp);
    b.classList.toggle("on", on);
    b.textContent = on ? "✓ nel confronto" : "+ confronta";
  }
  const bar = document.getElementById("cmpbar");
  if (!l.length || currentView() === "compare") { bar.hidden = true; document.body.classList.remove("has-cmpbar"); return; }
  bar.hidden = false;
  document.body.classList.add("has-cmpbar");
  bar.replaceChildren(h("div", { class: "wrap cmpbar-inner" },
    h("span", { class: "muted small-hide" }, "Confronto:"),
    h("div", { class: "cmpbar-chips" }, l.map((id) => h("span", { class: "chip" }, S.models[id].name,
      h("button", { class: "x", type: "button", "aria-label": `Togli ${S.models[id].name}`, onclick: () => cmpToggle(id) }, "✕")))),
    h("a", { class: "btn primary", href: "#v=compare" }, `Confronta (${l.length})`)));
}

/* ───────────── componenti ───────────── */
// stato di disponibilità: annunciato → repo senza pesi → pesi gated → pesi scaricabili (+ quantizzazioni pronte)
function availability(e) {
  if (e.kind === "closed") return { k: "closed", label: "Solo API", tip: "Modello proprietario: si usa solo via API" };
  if (e.kind === "announced") return { k: "announced", label: "Annunciato", tip: "Annunciato: né pesi né API rilevati finora" };
  if (e.kind === "placeholder") return { k: "placeholder", label: "Repo senza pesi", tip: "Il repository esiste ma non contiene ancora i file del modello" };
  if (e.gated) return { k: "gated", label: "Pesi gated", tip: "Pesi su Hugging Face con accesso limitato: si scaricano dopo aver accettato le condizioni sulla pagina del modello" };
  return { k: "open", label: "Pesi aperti", tip: "Pesi scaricabili liberamente da Hugging Face" };
}
function quantFormats(e) {
  const f = new Set((e.quants || []).map((q) => q.format));
  if ((e.quants || []).some((q) => q.nvfp4)) f.add("NVFP4");
  return ["GGUF", "NVFP4", "FP8", "MXFP4", "AWQ", "GPTQ", "MLX"].filter((x) => f.has(x));
}
const MODE = { bf16: "GPU · senza quantizzare", quant: "GPU · quantizzato", offload: "GPU + RAM (offload)", no: "Non supportato" };
const MODE_SHORT = { bf16: "BF16", quant: "quantizzato", offload: "GPU + RAM", no: "non entra" };
function kindBadge(e) { const a = availability(e); return h("span", { class: `badge ${a.k}`, title: a.tip }, a.label); }
function newBadge(e, ref) { return isNew(e, ref) ? h("span", { class: "badge new" }, "Nuovo") : null; }

function modelChips(e) {
  const c = [];
  if (e.params_b) {
    c.push(h("span", { class: "chip", title: e.params_source ? `fonte: ${e.params_source}` : null },
      e.active_b ? ["MoE ", h("b", {}, fmtParams(e.params_b)), ` · ${fmtParams(e.active_b)} attivi`] : h("b", {}, fmtParams(e.params_b))));
  }
  if (e.context) c.push(h("span", { class: "chip" }, `ctx ${fmtCtx(e.context)}`));
  if (e.pipeline) c.push(h("span", { class: "chip" }, e.pipeline));
  else if (e.modality) c.push(h("span", { class: "chip" }, e.modality));
  if (e.license) c.push(h("span", { class: "chip" }, e.license));
  if (e.pricing && e.pricing.in != null) c.push(h("span", { class: "chip", title: "$ per milione di token (input / output)" }, `$${fmtPrice(e.pricing.in)} / $${fmtPrice(e.pricing.out)}`));
  if (e.free_on_openrouter) c.push(h("span", { class: "chip" }, "gratis su OpenRouter"));
  const qf = quantFormats(e);
  if (qf.length && e.kind !== "closed") c.push(h("span", { class: "chip qready", title: "Quantizzazioni già pronte su Hugging Face" }, "pronte: ", h("b", {}, qf.slice(0, 3).join(" · "))));
  if (e.aa && e.aa.intelligence != null) c.push(h("span", { class: "chip aa", title: "Artificial Analysis Intelligence Index" }, "AA ", h("b", {}, fmtNum(e.aa.intelligence))));
  return h("div", { class: "chips" }, c);
}

function verdict(e) {
  const v = e.vram;
  if (v) return h("div", { class: `verdict ${v.level}` }, h("i", { class: "dotv" }),
    h("span", {}, v.mode ? h("b", { class: "mode" }, MODE[v.mode]) : null, v.text,
      h("span", { class: "est", title: "Calcolata da AI Radar dall'architettura del modello (config.json) e dall'hardware: è una stima, non una misura" }, " · stima")));
  if (e.kind === "closed") {
    const p = e.pricing && e.pricing.in != null ? `$${fmtPrice(e.pricing.in)} in / $${fmtPrice(e.pricing.out)} out per M token` : "prezzo n/d";
    return h("div", { class: "verdict api" }, h("i", { class: "dotv" }), h("span", {}, `Solo API · ${p}`));
  }
  if (e.kind === "announced") return h("div", { class: "verdict none" }, h("i", { class: "dotv" }), h("span", {}, "Annunciato: pesi/API non ancora rilevati"));
  if (e.kind === "open" && !e.params_b) return h("div", { class: "verdict none" }, h("i", { class: "dotv" }), h("span", {}, "Pesi disponibili, ma numero di parametri non determinato: niente stima della VRAM"));
  if (e.kind === "placeholder") return h("div", { class: "verdict no" }, h("i", { class: "dotv" }), h("span", {}, "Repository senza pesi"));
  return null;
}
function newerNote(e) {
  if (!e.newer) return null;
  const n = e.newer;
  const inReg = S.models[n.id];
  return h("p", { class: "newer" }, "⚠ Esiste una versione più recente: ",
    inReg ? h("a", { href: openModelHash(n.id) }, n.name) : ext(n.url, n.name), n.released ? ` (${fmtShort(n.released)})` : "");
}

function openModelHash(id) {
  const p = new URLSearchParams(location.hash.slice(1));
  p.set("m", id);
  return "#" + p.toString();
}

function modelCard(e, ref, tag) {
  const desc = (e.deep && e.deep.tldr) || e.description;
  const open = () => { location.hash = openModelHash(e.id); };
  const fresh = S.fresh.has(e.id);
  return h("article", { class: "mcard" + (fresh ? " fresh" : ""), tabindex: "0", role: "button", "aria-label": `Dettagli ${e.name}`, onclick: open,
      onkeydown: (ev) => { if (ev.key === "Enter" && ev.target === ev.currentTarget) open(); } },
    fresh ? h("span", { class: "fresh-tag" }, "novità per te") : null,
    h("div", { class: "mtop" }, kindBadge(e), newBadge(e, ref),
      h("span", { class: "muted" }, [e.org, fmtShort(e.released || e.first_seen)].filter(Boolean).join(" · ")),
      h("span", { class: "spacer" }), cmpBtn(e)),
    tag ? h("div", { class: "ntag " + tag.k }, tag.k === "version" ? ["Nuova versione di ", S.models[tag.of.id] ?
      h("a", { href: openModelHash(tag.of.id), onclick: (ev) => ev.stopPropagation() }, tag.of.name) : tag.of.name] : tag.label) : null,
    h("h3", {}, e.name),
    modelChips(e),
    verdict(e),
    e.deep && e.deep.hw_claims && e.vram && ["offload", "no"].includes(e.vram.mode) ?
      h("p", { class: "claim small" }, h("b", {}, "Il produttore dichiara: "), e.deep.hw_claims.length > 160 ? e.deep.hw_claims.slice(0, 157) + "…" : e.deep.hw_claims) : null,
    newerNote(e),
    desc ? h("p", { class: "tldr" }, desc) : null,
    e.note ? h("p", { class: "note" }, e.note) : null);
}

function newsItem(it) {
  const relCls = it.rel >= 8 ? "r-high" : it.rel >= 6 ? "r-mid" : "";
  const links = [];
  if (it.comments_url && it.comments_url !== it.url) links.push(ext(it.comments_url, { class: "mlink" }, "discussione"));
  for (const [k, u] of Object.entries(it.links || {})) links.push(ext(u, { class: "mlink" }, k === "arxiv" ? "arXiv" : k === "github" ? "GitHub" : "progetto"));
  const shown = new Set();
  for (const id of it.models || []) {
    const m = S.models[id];
    if (!m || shown.has(m.id)) continue; // alias di schede fuse e duplicati: un link per modello
    shown.add(m.id);
    links.push(h("a", { class: "mlink", href: openModelHash(m.id) }, "⬡ ", m.name));
  }
  const meta = [h("span", { class: `cat cat-${it.cat}` }, CAT1[it.cat] || it.cat),
    h("span", {}, it.source + (it.also_on && it.also_on.length ? ` +${it.also_on.length}` : ""))];
  if (it.engagement) meta.push(h("span", {}, it.source === "GitHub" ? `★ ${it.engagement}` : `${it.engagement} punti`));
  if (it.published) meta.push(h("span", {}, ago(it.published)));
  const fresh = S.fresh.has(it.id);
  return h("article", { class: "item" + (fresh ? " fresh" : "") },
    fresh ? h("span", { class: "fresh-tag" }, "novità per te") : null,
    h("div", { class: `rel ${relCls}`, title: it.llm ? "Rilevanza per te (0-10)" : "Punteggio euristico (senza LLM)" }, it.rel),
    h("div", { class: "body" },
      h("div", { class: "meta" }, meta),
      h("h3", {}, ext(it.url, it.title)),
      it.title !== it.title_orig ? h("div", { class: "orig" }, it.title_orig) : null,
      it.summary ? h("p", {}, it.summary) : null,
      it.why ? h("p", { class: "why" }, "→ ", it.why) : null,
      links.length ? h("div", { class: "links" }, links) : null));
}

function topicLabel(id) {
  const t = ((S.papers && S.papers.topics) || []).find((x) => x.id === id);
  return t ? t.label : id;
}
/* stato personale dei paper (salvati / letti), solo in questo browser */
const PST_KEY = "radar-papers-state";
function pState() { try { const x = JSON.parse(store(PST_KEY) || "null"); return x && x.saved ? x : { saved: {}, read: {} }; } catch { return { saved: {}, read: {} }; } }
function pToggle(kind, id) {
  const st = pState();
  if (st[kind][id]) delete st[kind][id]; else st[kind][id] = new Date().toISOString().slice(0, 10);
  store(PST_KEY, JSON.stringify(st));
  render();
}
const SCORE_HELP = "Rilevanza per te (0-10): quanto il paper è vicino ai tuoi argomenti e al tuo profilo e quanto puoi usarlo.\n" +
  "Valore scientifico (0-10): novità, rigore e portata nel suo campo, indipendentemente da te " +
  "(9-10 risultato di rilievo, 7-8 contributo solido, 4-6 incrementale, 0-3 marginale).\n" +
  "L'ordine «più rilevanti» combina i due voti con i voti su Hugging Face e la freschezza. Voti assegnati da un LLM sull'abstract.";
async function ensurePaperDetails() {
  if (S.pDetails) return;
  S.pDetails = (await getJSON("data/paper_details.json").catch(() => ({ papers: {} }))).papers || {};
}
function cachedOv(id) { try { return JSON.parse(store("radar-ov-" + id) || "null"); } catch { return null; } }
async function askPaperOverview(p, abs) {
  const sys = "Sei un divulgatore scientifico rigoroso: spieghi la ricerca AI in italiano semplice, senza gergo non spiegato " +
    "(chiarisci i termini tecnici tra parentesi), senza aggiungere informazioni che non sono nel testo. Rispondi SOLO con JSON valido.";
  const user = `Paper: ${p.title}\n\n${abs}\n\nRestituisci {"problema": "1-2 frasi", "idea": "2-3 frasi sul metodo e l'intuizione chiave", ` +
    '"risultati": "1-2 frasi con i numeri del testo", "limiti": "1 frase o stringa vuota", "glossario": [{"termine": "...", "spiegazione": "max 20 parole"}]}';
  const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST", headers: { Authorization: `Bearer ${getKey()}`, "Content-Type": "application/json", "X-Title": "AI Radar (personal)" },
    body: JSON.stringify({ model: S.index.site.advice_model, temperature: 0.2, max_tokens: 1200, response_format: { type: "json_object" },
      messages: [{ role: "system", content: sys }, { role: "user", content: user }] }),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error((d.error && d.error.message) || `HTTP ${r.status}`);
  const m = ((((d.choices || [])[0] || {}).message || {}).content || "").replace(/```(json)?/g, "").match(/\{[\s\S]*\}/);
  if (!m) throw new Error("risposta non valida");
  const x = JSON.parse(m[0]);
  return { problema: x.problema || "", idea: x.idea || "", risultati: x.risultati || "", limiti: x.limiti || "",
    glossario: (x.glossario || []).filter((g) => g && g.termine).slice(0, 3).map((g) => ({ t: g.termine, d: g.spiegazione || "" })), live: true };
}
function paperOverview(p) {
  if (!S.pDetails) {
    ensurePaperDetails().then(render);
    return h("div", { class: "p-ov" }, h("p", { class: "muted" }, "Carico i dettagli…"));
  }
  const det = S.pDetails[p.id] || {};
  const ov = det.ov || cachedOv(p.id);
  const abs = det.abs || "";
  const sec = (t, x) => x ? h("div", { class: "ov-sec" }, h("h4", {}, t), h("p", {}, x)) : null;
  const out = [];
  if (ov) {
    const vague = (x) => !x || /^(il testo |l'abstract )?(non (sono |vengono )?(specificat|riport|indicat|menzionat|dichiarat)|nessun limite)/i.test(x.trim());
    out.push(sec("Il problema", ov.problema), sec("L'idea", ov.idea), sec("I risultati", ov.risultati), vague(ov.limiti) ? null : sec("Limiti e dubbi", ov.limiti));
    if ((ov.glossario || []).length) out.push(h("div", { class: "ov-sec" }, h("h4", {}, "Parole chiave"),
      h("dl", { class: "gloss" }, ov.glossario.map((g) => [h("dt", {}, g.t), h("dd", {}, g.d)]))));
    out.push(h("p", { class: "small" }, ov.live ? `Spiegazione generata da ${S.index.site.advice_model} con il tuo OpenRouter.` : "Overview generata da un LLM dall'abstract: verifica i dettagli sul paper."));
  } else if (abs) {
    const box = h("div", {});
    if (getKey()) {
      const btn = h("button", { class: "btn small primary", type: "button" }, "Spiegamelo in modo semplice");
      btn.addEventListener("click", async () => {
        btn.disabled = true; btn.textContent = "Sto scrivendo la spiegazione…";
        try { const x = await askPaperOverview(p, abs); store("radar-ov-" + p.id, JSON.stringify(x)); render(); }
        catch (err) { box.replaceChildren(h("p", { class: "note" }, `Errore: ${err.message}`)); }
      });
      box.append(btn, h("span", { class: "small" }, ` ${S.index.site.advice_model} · ~0,001 $`));
    } else {
      box.append(h("p", { class: "small" }, "Overview non ancora pronta: la genero ogni giorno per i paper più rilevanti. ",
        "Per averla subito collega OpenRouter dalle impostazioni (⚙)."));
    }
    out.push(box);
  } else {
    out.push(h("p", { class: "muted small" }, "Dettagli non disponibili: apri il paper dal titolo."));
  }
  if (abs) out.push(h("details", { class: "abs" }, h("summary", {}, "Abstract originale (inglese)"), h("p", {}, abs)));
  return h("div", { class: "p-ov" }, out);
}
function paperCard(p) {
  const imp = p.imp != null ? p.imp : "–";
  const relCls = p.imp >= 8 ? "r-high" : p.imp >= 6 ? "r-mid" : "";
  const L = p.links || {};
  const st = pState();
  const saved = !!st.saved[p.id], read = !!st.read[p.id];
  const open = S.pOpen.has(p.id);
  const toggle = () => { if (S.pOpen.has(p.id)) S.pOpen.delete(p.id); else S.pOpen.add(p.id); render(); };
  const links = [L.arxiv && ext(L.arxiv, { class: "mlink" }, "arXiv"), L.pdf && ext(L.pdf, { class: "mlink" }, "PDF"),
    L.github && ext(L.github, { class: "mlink" }, "GitHub"), L.project && ext(L.project, { class: "mlink" }, "progetto")].filter(Boolean);
  // date distinte: pubblicazione originale, comparsa su HF Papers, rilevamento da parte di AI Radar
  const d10 = (x) => (x || "").slice(0, 10);
  const dates = [];
  if (p.published) dates.push(["pubblicato", d10(p.published)]);
  if (p.hf_date && d10(p.hf_date) !== d10(p.published)) dates.push(["su HF Papers", d10(p.hf_date)]);
  if (p.first_seen && !dates.some(([, d]) => d === p.first_seen)) dates.push(["rilevato", p.first_seen]);
  const meta = [h("span", {}, p.source), dates.length ? h("span", { title: "pubblicazione originale · comparsa su Hugging Face Papers · rilevamento da AI Radar" },
      dates.map(([k, d]) => `${k} ${fmtShort(d)}`).join(" · ")) : null,
    p.sci != null ? h("span", { class: "sci", title: SCORE_HELP }, `valore scientifico ${p.sci}/10`) : null,
    p.upvotes ? h("span", { title: "voti su Hugging Face Papers" }, `▲ ${p.upvotes}`) : null,
    p.authors && p.authors.length ? h("span", { class: "authors" }, p.authors.slice(0, 3).join(", ") + (p.authors.length > 3 ? " et al." : "")) : null,
    p.org ? h("span", {}, p.org) : null];
  const fresh = S.fresh.has("p:" + p.id);
  return h("article", { class: "item paper" + (fresh ? " fresh" : "") + (open ? " open" : "") + (read ? " read" : ""),
      onclick: (ev) => { if (!ev.target.closest("a, button, summary, details, dl")) toggle(); } },
    fresh ? h("span", { class: "fresh-tag" }, "novità per te") : null,
    h("div", { class: `rel ${relCls}`, title: SCORE_HELP }, imp),
    h("div", { class: "body" },
      h("div", { class: "meta" }, meta),
      h("h3", {}, ext(p.url, p.title_it || p.title)),
      p.title_it && p.title_it !== p.title ? h("div", { class: "orig" }, p.title) : null,
      p.summary_it ? h("p", {}, p.summary_it) : null,
      p.why ? h("p", { class: "why" }, "→ ", p.why) : null,
      open ? paperOverview(p) : null,
      h("div", { class: "links" },
        h("button", { class: "btn small" + (open ? " primary" : ""), type: "button", "aria-expanded": open ? "true" : "false", onclick: toggle },
          open ? "Chiudi" : p.has_ov ? "Overview ▾" : "Dettagli ▾"),
        h("button", { class: "tchip" + (saved ? " on" : ""), type: "button", title: "Salva per dopo", onclick: () => pToggle("saved", p.id) }, saved ? "★ salvato" : "☆ salva"),
        h("button", { class: "tchip" + (read ? " on" : ""), type: "button", title: "Segna come letto", onclick: () => pToggle("read", p.id) }, read ? "✓ letto" : "letto?"),
        (p.topics || []).map((t) => h("button", { class: "tchip" + (S.topics && S.topics.has(t) ? " on" : ""), type: "button",
          onclick: () => { S.topics = new Set([t]); saveTopics(); location.hash = "#v=papers"; render(); } }, topicLabel(t))), links)));
}

/* ───────────── novità dall'ultima visita ───────────── */
const SEEN_KEY = "radar-seen-v1";
function loadSeen() { try { const s = JSON.parse(store(SEEN_KEY) || "null"); return s && s.ids ? s : null; } catch { return null; } }
function dayIds(d) {
  const ids = d.items.map((i) => i.id).concat(d.models || [], (d.models_superseded || []).map((s) => s.id));
  if (S.papers) for (const p of Object.values(S.papers.papers)) if (p.first_seen === d.date) ids.push("p:" + p.id);
  return ids;
}
function trackFresh(d) {
  // eseguito una volta per giorno visualizzato: ciò che non avevi mai visto resta evidenziato per tutta la sessione
  if (S.freshDone[d.date]) return;
  S.freshDone[d.date] = true;
  const ids = dayIds(d);
  if (S.seenStart) for (const id of ids) if (!S.seenStart.ids[id]) S.fresh.add(id);
  const s = loadSeen() || { ids: {} };
  const today = new Date().toISOString().slice(0, 10);
  for (const id of ids) s.ids[id] = s.ids[id] || today;
  const cutoff = new Date(Date.now() - 60 * 864e5).toISOString().slice(0, 10);
  for (const [k, v] of Object.entries(s.ids)) if (v < cutoff) delete s.ids[k];
  s.at = new Date().toISOString();
  store(SEEN_KEY, JSON.stringify(s));
}
function freshBanner(d, models, papers) {
  if (!S.seenStart) {
    return h("div", { class: "fresh-banner" }, h("b", {}, "Benvenuto. "), "Dalla prossima visita qui trovi evidenziato tutto ciò che è arrivato dopo il tuo ultimo passaggio.");
  }
  const nModels = models.filter((e) => S.fresh.has(e.id)).length;
  const nNews = d.items.filter((i) => S.fresh.has(i.id) && i.rel >= 6).length;
  const nNewsAll = d.items.filter((i) => S.fresh.has(i.id)).length;
  const nPapers = papers.filter((p) => S.fresh.has("p:" + p.id)).length;
  const last = S.seenStart.at ? ago(S.seenStart.at) : "";
  if (!nModels && !nNewsAll && !nPapers) {
    return h("div", { class: "fresh-banner calm" }, `Nessuna novità da ${last ? "quando sei passato " + last : "ultima visita"}: hai già visto tutto.`);
  }
  const part = (n, one, many) => n ? `${n} ${n === 1 ? one : many}` : null;
  return h("div", { class: "fresh-banner" },
    h("div", {}, h("b", {}, "Novità dall'ultima visita"), last ? h("span", { class: "muted" }, ` (${last})`) : "", ": ",
      [part(nModels, "modello", "modelli"), part(nNews, "notizia rilevante", "notizie rilevanti"), part(nPapers, "paper sui tuoi argomenti", "paper sui tuoi argomenti")]
        .filter(Boolean).join(" · ") || `${nNewsAll} notizie minori`, ". Sono evidenziate in viola."),
    h("button", { class: "btn small" + (S.onlyFresh ? " primary" : ""), type: "button", onclick: () => { S.onlyFresh = !S.onlyFresh; render(); } },
      S.onlyFresh ? "Mostra tutto" : "Mostra solo le novità"));
}

/* ───────────── viste ───────────── */
function grid(key, list, ref, limit = 9, tagOf) {
  const shown = S.show[key] ? list : list.slice(0, limit);
  return [h("div", { class: "mgrid" }, shown.map((e) => modelCard(e, ref, tagOf ? tagOf(e) : null))),
    list.length > limit ? h("div", { class: "more" }, h("button", { class: "btn", onclick: () => { S.show[key] = !S.show[key]; render(); } },
      S.show[key] ? "Mostra meno" : `Mostra tutti (${list.length})`)) : null];
}

function viewDay() {
  const d = S.day;
  if (!d) return h("p", { class: "empty" }, "Nessuna rassegna per questa data.");
  const days = S.index.days.map((x) => x.date);
  const i = days.indexOf(d.date);
  const prev = i >= 0 && i < days.length - 1 ? days[i + 1] : null;
  const next = i > 0 ? days[i - 1] : null;
  const out = [h("div", { class: "daynav" },
    h("a", { class: "btn" + (prev ? "" : " disabled"), href: prev ? `#v=day&d=${prev}` : "#" }, "←", h("span", {}, " precedente")),
    h("h1", {}, fmtDate(d.date)),
    h("a", { class: "btn" + (next ? "" : " disabled"), href: next ? `#v=day&d=${next}` : "#" }, h("span", {}, "successivo "), "→"))];

  if (d.digest && d.digest.bullets && d.digest.bullets.length) {
    out.push(h("section", { class: "digest" },
      d.digest.headline ? h("h2", {}, d.digest.headline) : null,
      h("ul", {}, d.digest.bullets.map((b) => h("li", {}, b))),
      d.digest.gpu_note ? h("p", { class: "gpu-note" }, `🖥 ${S.index.hardware.gpu_name}: ${d.digest.gpu_note}`) : null,
      d.digest.by ? h("div", { class: "by" }, `Sintesi generata da ${d.digest.by}`) : null));
  }

  // modelli: solo le ultime versioni (un modello superato da uno più recente della stessa famiglia non compare)
  const byId = (ids) => (ids || []).map((id) => S.models[id]).filter((e) => e && !e.newer);
  let newModels = byId(d.models_new), cited = byId(d.models_cited);
  if (!d.models_new) { const all = byId(d.models); newModels = all.filter((e) => isNew(e, d.date)); cited = all.filter((e) => !isNew(e, d.date)); }
  const sup = [...(d.models_superseded || []), ...(d.models || []).map((id) => S.models[id]).filter((e) => e && e.newer)
    .map((e) => ({ id: e.id, name: e.name, newer: e.newer }))].filter((x, i, a) => a.findIndex((y) => y.id === x.id) === i);
  const dayPapers = S.papers ? Object.values(S.papers.papers).filter((p) => p.first_seen === d.date &&
    (!S.topics.size || (p.topics || []).some((t) => S.topics.has(t)))).sort((a, b) => b.score - a.score) : [];
  if (S.onlyFresh) {
    newModels = newModels.filter((e) => S.fresh.has(e.id));
    cited = cited.filter((e) => S.fresh.has(e.id));
  }
  out.splice(1, 0, freshBanner(d, [...newModels, ...cited], dayPapers));

  // in evidenza: le 5 notizie più rilevanti, poi un riepilogo cliccabile per sezione
  const top = d.items.filter((i) => i.rel >= 7 && (!S.onlyFresh || S.fresh.has(i.id))).sort((a, b) => b.rel - a.rel || (b.engagement || 0) - (a.engagement || 0)).slice(0, 5);
  if (top.length) {
    out.push(h("section", { class: "section top5" }, h("div", { class: "section-head" }, h("h2", {}, "Da sapere oggi"), h("span", { class: "muted" }, "le notizie più rilevanti per te")),
      h("ol", {}, top.map((it) => h("li", {}, h("span", { class: `rel mini ${it.rel >= 8 ? "r-high" : "r-mid"}`, title: "Rilevanza per te (0-10)" }, it.rel),
        h("div", {}, ext(it.url, h("b", {}, it.title)), h("span", { class: "muted small" }, ` · ${it.source}`),
          it.why || it.summary ? h("div", { class: "small" }, it.why || it.summary) : null))))));
  }
  const qn = (d.quants_new || []).filter((q) => S.models[q.id]);
  const nTools = d.items.filter((i) => i.cat === "tool" || i.cat === "feature").length;
  const go = (id, fn) => () => { if (fn) fn(); render(); const el = document.getElementById(id); if (el) el.scrollIntoView({ behavior: "smooth", block: "start" }); };
  const tile = (n, label, sub, onclick) => h("button", { class: "tile", type: "button", onclick, disabled: !n }, h("b", {}, n), h("span", {}, label), sub ? h("span", { class: "small" }, sub) : null);
  out.push(h("div", { class: "tiles" },
    tile(newModels.length, "nuovi modelli", `${newModels.filter((e) => (d.model_tags || {})[e.id]).length} nuove versioni`, go("sec-new")),
    tile(qn.length, "nuove quantizzazioni", "GGUF, NVFP4, FP8…", go("sec-quants")),
    tile(dayPapers.length, "paper sui tuoi argomenti", null, go("sec-papers")),
    tile(nTools, "tool e feature", null, go("sec-news", () => { S.cat = "tool"; })),
    tile(d.items.length, "notizie in tutto", null, go("sec-news", () => { S.cat = "tutte"; S.show.items = true; }))));

  const tagOf = (e) => (d.model_tags || {})[e.id] ? { k: "version", of: d.model_tags[e.id] } : null;
  if (newModels.length) {
    out.push(h("section", { class: "section", id: "sec-new" },
      h("div", { class: "section-head" }, h("h2", {}, "Nuovi modelli"), h("span", { class: "muted" }, `${newModels.length} · usciti negli ultimi ${S.index.new_model_days || 14} giorni, i più rilevanti per primi`)),
      grid("new", newModels, d.date, 6, (e) => tagOf(e) || { k: "model", label: "Nuovo modello" })));
  }
  if (qn.length && !S.onlyFresh) {
    out.push(h("section", { class: "section", id: "sec-quants" },
      h("div", { class: "section-head" }, h("h2", {}, "Nuove quantizzazioni"), h("span", { class: "muted" }, "modelli già noti che da oggi hanno un formato pronto in più")),
      h("ul", { class: "qnew" }, qn.map((q) => h("li", {}, h("a", { href: openModelHash(q.id) }, S.models[q.id].name), ": ora anche in ",
        q.formats.map((f, i) => [i ? ", " : "", q.repos && q.repos[f] ? ext(`https://huggingface.co/${q.repos[f]}`, h("b", {}, f)) : h("b", {}, f)]),
        S.models[q.id].vram ? h("span", { class: "small" }, ` · ${S.models[q.id].vram.label}`) : null)))));
  }
  if (cited.length) {
    out.push(h("section", { class: "section" },
      h("div", { class: "section-head" }, h("h2", {}, "Citati oggi"), h("span", { class: "muted" }, "modelli già usciti di cui si parla nelle notizie (semplice menzione)")),
      grid("cited", cited, d.date, 6)));
  }
  if (sup.length && !S.onlyFresh) {
    out.push(h("p", { class: "small superseded-note" }, `Nascosti ${sup.length} modelli citati oggi perché esiste già una versione più recente: `,
      sup.map((s, i) => [i ? " · " : "", h("a", { href: openModelHash(s.id) }, s.name), " → ",
        S.models[s.newer.id] ? h("a", { href: openModelHash(s.newer.id) }, s.newer.name) : ext(s.newer.url, s.newer.name)])));
  }

  // paper del giorno per gli argomenti scelti
  if (S.papers) {
    const day = dayPapers.filter((p) => !S.onlyFresh || S.fresh.has("p:" + p.id)).slice(0, 6);
    if (day.length) {
      out.push(h("section", { class: "section", id: "sec-papers" },
        h("div", { class: "section-head" }, h("h2", {}, "Paper per i tuoi argomenti"),
          h("span", { class: "muted" }, [...S.topics].map(topicLabel).join(", ") || "tutti gli argomenti"),
          h("a", { href: "#v=papers", class: "head-link" }, "Tutti i paper →")),
        h("div", { class: "items" }, day.map(paperCard))));
    }
  }

  // notizie
  const counts = {};
  for (const it of d.items) counts[it.cat] = (counts[it.cat] || 0) + 1;
  const q = S.q.trim().toLowerCase();
  let items = d.items.filter((it) => (S.cat === "tutte" || it.cat === S.cat) && it.rel >= S.minRel && (!S.onlyFresh || S.fresh.has(it.id)) &&
    (!q || `${it.title} ${it.title_orig} ${it.summary} ${it.source}`.toLowerCase().includes(q)));
  const total = items.length;
  if (!S.show.items) items = items.slice(0, 15);
  const chip = (key, label, n) => h("button", { class: "fchip" + (S.cat === key ? " on" : ""), onclick: () => { S.cat = key; S.show.items = false; render(); } },
    label, n != null ? h("span", { class: "n" }, n) : null);
  out.push(h("section", { class: "section", id: "sec-news" },
    h("div", { class: "section-head" }, h("h2", {}, "Rassegna completa"), h("span", { class: "muted" }, `${d.items.length} notizie selezionate su ${d.stats ? d.stats.new : "?"} nuove, dalla più rilevante`)),
    h("div", { class: "filters" },
      chip("tutte", "Tutte", d.items.length),
      Object.keys(CATS).filter((c) => counts[c]).map((c) => chip(c, CATS[c], counts[c])),
      h("select", { class: "search", "aria-label": "Rilevanza minima", onchange: (ev) => { S.minRel = +ev.target.value; render(); } },
        [[0, "Ogni rilevanza"], [5, "Rilevanza ≥ 5"], [7, "Rilevanza ≥ 7"]].map(([v, l]) => h("option", { value: v, selected: S.minRel === v }, l))),
      h("input", { class: "search", type: "search", placeholder: "Cerca…", value: S.q, "aria-label": "Cerca nelle notizie",
        oninput: (ev) => { S.q = ev.target.value; renderKeepFocus(ev.target); } })),
    items.length ? h("div", { class: "items" }, items.map(newsItem)) : h("p", { class: "empty" }, "Nessuna notizia con questi filtri."),
    total > items.length ? h("div", { class: "more" }, h("button", { class: "btn", onclick: () => { S.show.items = true; render(); } }, `Mostra tutte (${total})`)) : null));
  return out;
}

function latestToggle(hidden) {
  return h("label", { class: "toggle", title: "Nasconde i modelli per cui esiste già una versione più recente della stessa famiglia" },
    h("input", { type: "checkbox", checked: S.latestOnly, onchange: (ev) => { S.latestOnly = ev.target.checked; render(); } }),
    ` Solo ultime versioni${hidden ? ` (${hidden} nascosti)` : ""}`);
}

function viewModels() {
  const all = Object.values(S.models);
  const q = S.mQ.trim().toLowerCase();
  const hidden = all.filter((e) => e.newer).length;
  let list = all.filter((e) => {
    if (S.latestOnly && e.newer) return false;
    if (S.mFilter === "open" && e.kind !== "open") return false;
    if (S.mFilter === "api" && e.kind !== "closed" && !e.or_id) return false;
    if (S.mFilter === "gpu" && !(e.vram && e.vram.level === "ok")) return false;
    if (S.mFilter === "new" && !isNew(e)) return false;
    if (S.mFilter === "announced" && !["announced", "placeholder"].includes(e.kind)) return false;
    return !q || `${e.name} ${e.org} ${e.id} ${e.pipeline || ""}`.toLowerCase().includes(q);
  });
  const sorters = {
    recent: (a, b) => (b.released || b.first_seen || "").localeCompare(a.released || a.first_seen || ""),
    score: (a, b) => (b.score || 0) - (a.score || 0),
    aa: (a, b) => ((b.aa || {}).intelligence || -1) - ((a.aa || {}).intelligence || -1),
    params: (a, b) => (b.params_b || 0) - (a.params_b || 0),
    likes: (a, b) => ((b.signals || {}).likes || 0) - ((a.signals || {}).likes || 0),
  };
  list.sort(sorters[S.mSort]);
  const total = list.length;
  list = list.slice(0, S.mLimit);
  const chip = (key, label) => h("button", { class: "fchip" + (S.mFilter === key ? " on" : ""), onclick: () => { S.mFilter = key; S.mLimit = 60; render(); } }, label);
  return [
    h("div", { class: "section-head" }, h("h2", {}, "Registro modelli"), h("span", { class: "muted" }, `${all.length} modelli tracciati`)),
    h("div", { class: "filters" },
      chip("all", "Tutti"), chip("new", "Nuovi"), chip("gpu", `Girano sulla ${S.index.hardware.gpu_name}`), chip("open", "Pesi aperti"), chip("api", "API"), chip("announced", "Annunciati / senza pesi"),
      h("select", { class: "search", "aria-label": "Ordina", onchange: (ev) => { S.mSort = ev.target.value; render(); } },
        [["recent", "Più recenti"], ["aa", "Indice AA"], ["score", "Più rilevanti"], ["likes", "Più popolari"], ["params", "Più grandi"]].map(([v, l]) => h("option", { value: v, selected: S.mSort === v }, l))),
      h("input", { class: "search", type: "search", placeholder: "Cerca modello…", value: S.mQ, "aria-label": "Cerca modello",
        oninput: (ev) => { S.mQ = ev.target.value; renderKeepFocus(ev.target); } }),
      latestToggle(hidden)),
    list.length ? h("div", { class: "mgrid" }, list.map((e) => modelCard(e))) : h("p", { class: "empty" }, "Nessun modello."),
    total > list.length ? h("div", { class: "more" }, h("button", { class: "btn", onclick: () => { S.mLimit += 60; render(); } }, `Altri (${total - list.length})`)) : null,
  ];
}

function viewLeaderboard() {
  const meta = S.index.benchmarks_meta || {};
  const gpu = S.index.hardware.gpu_name;
  const q = S.lbQ.trim().toLowerCase();
  // Anche i modelli che Artificial Analysis elenca ma senza indici (celle «n.v.»): vanno in fondo, ma restano confrontabili e filtrabili
  let rows = Object.values(S.models).filter((e) => e.aa);
  const total = rows.length;
  const scored = rows.filter((e) => e.aa.intelligence != null || e.aa.coding != null || e.aa.agentic != null).length;
  const hidden = rows.filter((e) => e.newer).length;
  rows = rows.filter((e) => {
    if (S.latestOnly && e.newer) return false;
    if (S.lbFilter === "open" && !(e.hf_id || e.kind === "open")) return false;
    if (S.lbFilter === "api" && e.kind !== "closed") return false;
    if (S.lbFilter === "gpu" && !(e.vram && e.vram.level === "ok")) return false;
    if (S.lbFilter === "new" && !isNew(e)) return false;
    return !q || `${e.name} ${e.org}`.toLowerCase().includes(q);
  });
  const val = {
    intelligence: (e) => e.aa.intelligence, coding: (e) => e.aa.coding, agentic: (e) => e.aa.agentic,
    price: (e) => e.pricing ? e.pricing.out : null, context: (e) => e.context, released: (e) => e.released || "", name: (e) => e.name.toLowerCase(),
  };
  const f = val[S.lbSort];
  rows.sort((a, b) => {
    const x = f(a), y = f(b);
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * S.lbDir;
  });
  const maxI = Math.max(...rows.map((e) => e.aa.intelligence || 0), 1);
  const th = (key, label, cls) => h("th", { class: (cls || "") + " sortable" + (S.lbSort === key ? " sorted" : ""), scope: "col",
    onclick: () => { if (S.lbSort === key) S.lbDir *= -1; else { S.lbSort = key; S.lbDir = ["price", "name"].includes(key) ? 1 : -1; } render(); } },
    label, S.lbSort === key ? (S.lbDir < 0 ? " ↓" : " ↑") : "");
  const chip = (key, label) => h("button", { class: "fchip" + (S.lbFilter === key ? " on" : ""), onclick: () => { S.lbFilter = key; render(); } }, label);
  const NV = () => h("span", { class: "nv", title: "Non valutato da Artificial Analysis su questo indice: non significa punteggio basso" }, "n.v.");
  const cell = (v) => v == null ? NV() : fmtNum(v);
  const bar = (v, max, cls) => v == null ? NV() :
    h("div", { class: "ibar " + (cls || "") }, h("i", { style: `width:${Math.max(2, v / max * 100)}%` }), h("b", {}, fmtNum(v)));
  return [
    h("div", { class: "section-head" }, h("h2", {}, "Classifica"),
      h("span", { class: "muted" }, `${total} modelli di Artificial Analysis, ${scored} con indici (gli altri sono in fondo, «n.v.») · clicca le colonne per ordinare`)),
    h("div", { class: "filters" },
      chip("all", "Tutti"), chip("open", "Pesi aperti"), chip("gpu", `Girano sulla ${gpu}`), chip("api", "Solo API"), chip("new", "Nuovi"),
      h("input", { class: "search", type: "search", placeholder: "Cerca…", value: S.lbQ, "aria-label": "Cerca in classifica",
        oninput: (ev) => { S.lbQ = ev.target.value; renderKeepFocus(ev.target); } }),
      latestToggle(hidden)),
    h("div", { class: "tbl-wrap card" }, h("table", { class: "tbl lb" },
      h("thead", {}, h("tr", {}, h("th", { class: "num" }, "#"), th("name", "Modello"), th("intelligence", "Intelligence", "wide"),
        th("coding", "Coding"), th("agentic", "Agentic"), th("price", "API $ in / out"), th("context", "Contesto"),
        h("th", {}, gpu), th("released", "Uscita"), h("th", {}, ""))),
      h("tbody", {}, rows.map((e, i) => h("tr", {},
        h("td", { class: "num muted" }, i + 1),
        h("td", {}, h("a", { href: openModelHash(e.id), class: "lb-name" }, e.name), h("div", { class: "small" }, kindBadge(e), " ", e.org || "", newBadge(e) ? " · nuovo" : "")),
        h("td", { class: "wide" }, bar(e.aa.intelligence, maxI)),
        h("td", { class: "num" }, cell(e.aa.coding)), h("td", { class: "num" }, cell(e.aa.agentic)),
        h("td", { class: "num" }, e.pricing && e.pricing.in != null ? `${fmtPrice(e.pricing.in)} / ${fmtPrice(e.pricing.out)}` : "—"),
        h("td", { class: "num" }, fmtCtx(e.context) || "—"),
        h("td", {}, e.vram ? [h("span", { class: `lvl ${e.vram.level}`, title: vramTip(e.vram) }, e.vram.label),
          e.vram.mode && e.vram.mode !== "no" ? h("div", { class: "small" }, MODE_SHORT[e.vram.mode]) : null] : h("span", { class: "muted" }, e.kind === "closed" ? "solo API" : "—")),
        h("td", { class: "num" }, fmtShort(e.released) || "—"),
        h("td", {}, cmpBtn(e))))))),
    h("p", { class: "small source-note" },
      `Colonna ${gpu}: quantizzazione consigliata (la migliore che lascia almeno ${S.index.hardware.margin_gb ?? 1.5} GB liberi) e contesto massimo; `,
      "«BF16 nativo» = gira senza quantizzare; «GPU + RAM» = gira solo con parte dei pesi in RAM (lento). Passa il mouse per le alternative, clicca il modello per il calcolatore completo. ",
      "«n.v.» = indice non misurato per quel modello, non un punteggio basso. Prezzi: API via OpenRouter, $ per milione di token; in locale un modello aperto costa solo l'hardware. ",
      "Intelligence, Coding e Agentic sono gli indici compositi di Artificial Analysis, misurati da loro con la stessa metodologia per tutti i modelli: ",
      "a differenza dei benchmark dichiarati dai produttori, sono confrontabili tra loro. Per i modelli con più modalità di ragionamento è indicata la variante migliore. ",
      meta.citation ? meta.citation : "Fonte: Artificial Analysis via OpenRouter.", meta.as_of ? ` Dati al ${fmtShort(meta.as_of)}.` : ""),
  ];
}

function saveTopics() { store("radar-topics", JSON.stringify([...S.topics])); }
function viewPapers() {
  const P = S.papers;
  if (!P || !Object.keys(P.papers || {}).length) return h("p", { class: "empty" }, "Nessun paper ancora: arriveranno con la prossima esecuzione.");
  const cutoff = new Date(Date.now() - S.pDays * 864e5).toISOString().slice(0, 10);
  const pst = pState();
  const pWhen = (p) => [p.published, p.hf_date, p.first_seen].filter(Boolean).map((x) => x.slice(0, 10)).sort().pop() || "";
  const inWin = Object.values(P.papers).filter((p) => S.pView === "saved" ? pst.saved[p.id] : pWhen(p) >= cutoff)
    .filter((p) => S.pView !== "unread" || !pst.read[p.id]);
  const counts = {};
  for (const p of inWin) for (const t of p.topics || []) counts[t] = (counts[t] || 0) + 1;
  const q = S.pQ.trim().toLowerCase();
  let list = inWin.filter((p) => (!S.topics.size || (p.topics || []).some((t) => S.topics.has(t))) &&
    (!q || `${p.title} ${p.title_it || ""} ${p.summary_it || ""} ${(p.authors || []).join(" ")}`.toLowerCase().includes(q)));
  const sorters = { score: (a, b) => b.score - a.score, votes: (a, b) => (b.upvotes || 0) - (a.upvotes || 0) || b.score - a.score,
    sci: (a, b) => (b.sci ?? -1) - (a.sci ?? -1) || b.score - a.score, recent: (a, b) => pWhen(b).localeCompare(pWhen(a)) };
  list.sort(sorters[S.pSort]);
  const total = list.length;
  list = list.slice(0, S.pLimit);
  const tchip = (t) => h("button", { class: "fchip" + (S.topics.has(t.id) ? " on" : ""), type: "button", title: t.desc,
    onclick: () => { if (S.topics.has(t.id)) S.topics.delete(t.id); else S.topics.add(t.id); saveTopics(); S.pLimit = 40; render(); } },
    t.label, h("span", { class: "n" }, counts[t.id] || 0));
  return [
    h("div", { class: "section-head" }, h("h2", {}, "Paper"), h("span", { class: "muted" }, "scegli gli argomenti: la selezione resta salvata e vale anche per la pagina Oggi")),
    h("div", { class: "filters topics" }, P.topics.map(tchip),
      h("button", { class: "btn small", type: "button", onclick: () => { S.topics = new Set(P.topics.map((t) => t.id)); saveTopics(); render(); } }, "tutti"),
      h("button", { class: "btn small", type: "button", onclick: () => { S.topics = new Set(); saveTopics(); render(); } }, "nessuno")),
    h("div", { class: "filters" },
      h("select", { class: "search", "aria-label": "Periodo", onchange: (ev) => { S.pDays = +ev.target.value; render(); } },
        [[2, "Ultimi 2 giorni"], [7, "Ultima settimana"], [30, "Ultimo mese"], [60, "Ultimi 2 mesi"]].map(([v, l]) => h("option", { value: v, selected: S.pDays === v }, l))),
      h("select", { class: "search", "aria-label": "Ordina paper", onchange: (ev) => { S.pSort = ev.target.value; render(); } },
        [["score", "Più rilevanti per te"], ["sci", "Valore scientifico"], ["votes", "Più votati su HF"], ["recent", "Più recenti"]].map(([v, l]) => h("option", { value: v, selected: S.pSort === v }, l))),
      [["all", "Tutti"], ["unread", "Da leggere"], ["saved", `Salvati (${Object.keys(pst.saved).filter((id) => P.papers[id]).length})`]].map(([k, l]) =>
        h("button", { class: "fchip" + (S.pView === k ? " on" : ""), type: "button", onclick: () => { S.pView = k; S.pLimit = 40; render(); } }, l)),
      h("input", { class: "search", type: "search", placeholder: "Cerca titolo, autore…", value: S.pQ, "aria-label": "Cerca nei paper",
        oninput: (ev) => { S.pQ = ev.target.value; renderKeepFocus(ev.target); } }),
      h("span", { class: "muted small" }, `${total} paper`)),
    list.length ? h("div", { class: "items" }, list.map(paperCard)) : h("p", { class: "empty" }, "Nessun paper con questi filtri."),
    total > list.length ? h("div", { class: "more" }, h("button", { class: "btn", onclick: () => { S.pLimit += 40; render(); } }, `Altri (${total - list.length})`)) : null,
    h("p", { class: "small source-note" }, "Fonti: Hugging Face Daily Papers, arXiv (ricerca per argomento), Alignment Forum. ",
      "Il numero nel riquadro è la rilevanza per te, accanto c'è il valore scientifico nel suo campo (passa il mouse per i criteri). ",
      "Clicca un paper per l'overview (problema, idea, risultati, limiti). Argomenti, voti e riassunti sono assegnati da un LLM: verifica sempre sul paper. ",
      "«Salvati» e «letti» restano solo in questo browser."),
  ];
}

async function searchArchive(q) {
  S.archQ = q;
  const qq = q.trim().toLowerCase();
  const inp = () => app.querySelector('input[aria-label="Cerca in tutto l\'archivio"]');
  if (qq.length < 3) { S.archRes = null; if (inp()) renderKeepFocus(inp()); else render(); return; }
  const missing = S.index.days.map((d) => d.date).filter((d) => !S.archDays[d]);
  await Promise.all(missing.map((d) => getJSON(`data/days/${d}.json`).then((x) => { S.archDays[d] = x; }).catch(() => { S.archDays[d] = { items: [] }; })));
  if (S.archQ !== q) return;
  const res = [];
  for (const d of S.index.days) {
    for (const it of (S.archDays[d.date] || {}).items || []) {
      if (`${it.title} ${it.title_orig} ${it.summary || ""}`.toLowerCase().includes(qq)) res.push({ date: d.date, it });
    }
  }
  S.archRes = res.sort((a, b) => b.date.localeCompare(a.date) || b.it.rel - a.it.rel).slice(0, 100);
  if (inp()) renderKeepFocus(inp()); else render();
}
function viewArchive() {
  const n = S.index.days.length;
  return [
    h("div", { class: "section-head" }, h("h2", {}, "Archivio"), h("span", { class: "muted" }, plural(n, "giorno", "giorni"))),
    h("div", { class: "filters" }, h("input", { class: "search wide", type: "search", placeholder: "Cerca in tutte le notizie dell'archivio (min. 3 lettere)…", value: S.archQ,
      "aria-label": "Cerca in tutto l'archivio", oninput: (ev) => searchArchive(ev.target.value) })),
    S.archRes ? h("section", { class: "section" }, h("div", { class: "section-head" }, h("h3", { class: "h3" }, `${S.archRes.length} risultati`)),
      S.archRes.length ? h("div", { class: "days" }, S.archRes.map(({ date, it }) => h("div", { class: "dayrow" },
        h("a", { class: "d", href: `#v=day&d=${date}` }, fmtShort(date)), h("span", { class: "hl" }, ext(it.url, it.title), it.summary ? h("div", { class: "small" }, it.summary) : null),
        h("span", { class: "c" }, `${CAT1[it.cat] || it.cat} · ril. ${it.rel}`)))) : h("p", { class: "empty" }, "Nessuna notizia trovata.")) : null,
    h("div", { class: "days" }, S.index.days.map((d) => h("a", { class: "dayrow", href: `#v=day&d=${d.date}` },
      h("span", { class: "d" }, fmtDate(d.date)), h("span", { class: "hl" }, d.headline || h("span", { class: "muted" }, "—")),
      h("span", { class: "c" }, `${plural(d.n_items, "notizia", "notizie")} · ${plural(d.n_models, "modello", "modelli")}`)))),
  ];
}

/* ───────────── embedding (leaderboard MTEB) ───────────── */
const EMB_TYPES = ["Retrieval", "Reranking", "STS", "Classification", "PairClassification", "Clustering", "MultilabelClassification",
  "BitextMining", "InstructionReranking", "Summarization"];
// nomi ufficiali MTEB (in tabella) e spiegazione (al passaggio del mouse)
const TYPE_IT = { Retrieval: "Retrieval", Reranking: "Reranking", STS: "STS", Classification: "Classification",
  PairClassification: "Pair Classification", Clustering: "Clustering", MultilabelClassification: "Multilabel Classification",
  BitextMining: "Bitext Mining", InstructionReranking: "Instruction Reranking", Summarization: "Summarization" };
const TYPE_HELP = { Retrieval: "trovare i documenti pertinenti a una query (RAG, ricerca)", Reranking: "riordinare candidati per pertinenza",
  STS: "similarità semantica tra frasi", Classification: "classificare testi con un classificatore sugli embedding",
  PairClassification: "decidere se due testi sono equivalenti/duplicati/parafrasi", Clustering: "raggruppare testi simili",
  MultilabelClassification: "classificazione con più etichette", BitextMining: "trovare frasi equivalenti in lingue diverse",
  InstructionReranking: "reranking che segue istruzioni", Summarization: "valutare riassunti" };
const LANG_IT = { Italian: "Italiano", English: "Inglese", German: "Tedesco", French: "Francese", Spanish: "Spagnolo" };
function embFits(m) {
  if (!m.params) return null;
  const hw = S.index.hardware;
  const gb = m.params * 1e9 * 2 / GiB + 1.5; // BF16 + buffer per batch di encoding
  return { gb, fits: gb <= hw.vram_gb - hw.reserved_gb };
}
const embNew = (m) => !!m.released && (Date.now() - new Date(m.released.slice(0, 10) + "T12:00:00")) / 864e5 <= 45;
function embScore(name) {
  const B = S.emb.benchmarks[0];
  const r = B.rows.find((x) => x.m === name);
  return r ? { label: B.label, mean: r.mean, rank: r.rank } : null;
}
function embCard(m, unbenchmarked) {
  const f = embFits(m);
  const sc = !unbenchmarked && embScore(m.name);
  const chip = (...k) => h("span", { class: "chip" }, ...k);
  return h("article", { class: "mcard ecard" },
    h("div", { class: "mtop" }, h("span", { class: `badge ${m.open ? "open" : "closed"}` }, m.open ? "Pesi aperti" : "API"),
      embNew(m) ? h("span", { class: "badge new" }, "Nuovo") : null,
      unbenchmarked ? h("span", { class: "badge announced" }, m.kind === "reranker" ? "reranker · non su MTEB" : "non ancora su MTEB") : null,
      h("span", { class: "muted" }, fmtShort(m.released))),
    h("h3", {}, ext(m.url || `https://huggingface.co/${m.name}`, m.name.split("/").pop())),
    h("div", { class: "small" }, m.name.includes("/") ? m.name.split("/")[0] : ""),
    h("div", { class: "chips" }, m.params ? chip(h("b", {}, fmtParams(m.params))) : null, m.dim ? chip(`dim ${m.dim}`) : null,
      m.maxTokens ? chip(`${fmtCtx(m.maxTokens)} token`) : null, m.lang ? chip(`${LANG_IT[S.emb.language] || S.emb.language} ✓`) : null,
      f && m.open ? h("span", { class: `chip lvl ${f.fits ? "ok" : "no"}` }, `${f.gb.toFixed(1)} GB ${f.fits ? "✓" : "✗"}`) : null),
    sc ? h("p", { class: "small" }, `MTEB ${sc.label}: `, h("b", {}, fmtNum(sc.mean)), ` · #${sc.rank}`) : null);
}

function viewEmbeddings() {
  const E = S.emb;
  if (!E || !E.benchmarks || !E.benchmarks.length) return h("p", { class: "empty" }, "Dati MTEB non ancora disponibili: arriveranno con la prossima esecuzione.");
  const B = E.benchmarks[Math.min(S.eBench, E.benchmarks.length - 1)];
  const present = new Set(B.rows.flatMap((r) => Object.keys(r.types).filter((t) => r.types[t] != null)));
  const types = EMB_TYPES.filter((t) => present.has(t));
  const colCls = (key, base) => (base || "num") + (S.eSort === key ? " sorted-col" : "");
  const langLabel = LANG_IT[E.language] || E.language;
  const q = S.eQ.trim().toLowerCase();
  const sizeOk = (p) => S.eSize === "all" || (p != null && (S.eSize === "s" ? p < 1 : S.eSize === "m" ? p >= 1 && p < 5 : S.eSize === "l" ? p >= 5 && p < 10 : p >= 10));
  let rows = B.rows.map((r) => ({ ...r, meta: E.models[r.m] || { name: r.m } }));
  const hiddenOld = rows.filter((r) => r.meta.newer).length;
  rows = rows.filter((r) => {
    const m = r.meta;
    if (S.latestOnly && m.newer) return false;
    if (S.eOpen && !m.open) return false;
    if (S.eVram !== "all") { const f = embFits(m); if (!m.open || !f || (S.eVram === "gpu" ? !f.fits : f.gb > +S.eVram)) return false; }
    if (S.eLang && !m.lang) return false;
    return sizeOk(m.params) && (!q || m.name.toLowerCase().includes(q));
  });
  const getv = { mean: (r) => r.mean, meanType: (r) => r.meanType, lang: (r) => r.lang, params: (r) => r.meta.params, dim: (r) => r.meta.dim,
    maxTokens: (r) => r.meta.maxTokens, released: (r) => r.meta.released || "", name: (r) => r.m.toLowerCase(), rank: (r) => r.rank };
  for (const t of types) getv["t:" + t] = (r) => r.types[t];
  const f = getv[S.eSort] || getv.mean;
  rows.sort((a, b) => {
    const x = f(a), y = f(b);
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * S.eDir;
  });
  const th = (key, label, title) => h("th", { class: "sortable" + (S.eSort === key ? " sorted" : ""), scope: "col", title,
    onclick: () => { if (S.eSort === key) S.eDir *= -1; else { S.eSort = key; S.eDir = ["name", "params", "rank"].includes(key) ? 1 : -1; } render(); } },
    label, S.eSort === key ? (S.eDir < 0 ? " ↓" : " ↑") : "");
  const toggle = (key, label) => h("button", { class: "fchip" + (S[key] ? " on" : ""), type: "button", onclick: () => { S[key] = !S[key]; render(); } }, label);
  const maxMean = Math.max(...rows.map((r) => r.mean || 0), 1);
  const recent = Object.values(E.models).filter((m) => embNew(m) && !m.newer).sort((a, b) => (b.released || "").localeCompare(a.released || ""));
  const unb = E.new_unbenchmarked || [];
  const gpu = S.index.hardware.gpu_name;
  return [
    h("div", { class: "section-head" }, h("h2", {}, "Embedding"), h("span", { class: "muted" }, `leaderboard MTEB ufficiale · ${Object.keys(E.models).length} modelli`)),
    recent.length || unb.length ? h("section", {},
      h("div", { class: "section-head" }, h("h3", { class: "h3" }, "Novità"), h("span", { class: "muted" }, "usciti negli ultimi 45 giorni")),
      h("div", { class: "mgrid" }, recent.slice(0, 6).map((m) => embCard(m, false)), unb.slice(0, 6).map((m) => embCard(m, true)))) : null,
    h("div", { class: "filters topics", style: "margin-top:22px" }, E.benchmarks.map((b, i) => h("button", { class: "fchip" + (i === S.eBench ? " on" : ""), type: "button", title: b.name,
      onclick: () => {
        S.eBench = i;
        const nb = E.benchmarks[i];
        const has = (t) => nb.rows.some((r) => r.types[t] != null);
        if ((S.eSort.startsWith("t:") && !has(S.eSort.slice(2))) || (S.eSort === "lang" && !nb.language)) { S.eSort = "rank"; S.eDir = 1; }
        render();
      } }, b.label))),
    h("div", { class: "filters task-pick" }, h("span", { class: "task-q" }, "Quale task ti serve?"),
      [["rank", "Rank ufficiale", "ordine della leaderboard MTEB"], ["mean", "Media", "media di tutti i task"],
        B.language ? ["lang", langLabel, `media dei task in ${langLabel.toLowerCase()}`] : null,
        ...types.map((t) => ["t:" + t, TYPE_IT[t], TYPE_HELP[t]])].filter(Boolean).map(([key, label, help]) =>
        h("button", { class: "fchip" + (S.eSort === key ? " on" : ""), type: "button", title: help,
          onclick: () => { S.eSort = key; S.eDir = key === "rank" ? 1 : -1; render(); } }, label)),
      h("span", { class: "small muted" }, S.eSort.startsWith("t:") ? `ordinati per il punteggio ${TYPE_IT[S.eSort.slice(2)]}, non per la media` : "")),
    h("div", { class: "filters" },
      toggle("eOpen", "Pesi aperti"), toggle("eLang", `Supportano l'${langLabel.toLowerCase()}`),
      h("select", { class: "search", "aria-label": "VRAM massima", onchange: (ev) => { S.eVram = ev.target.value; render(); } },
        [["all", "Ogni VRAM"], ["4", "≤ 4 GB"], ["8", "≤ 8 GB"], ["16", "≤ 16 GB"], ["gpu", `Entrano nella ${gpu}`]].map(([v, l]) => h("option", { value: v, selected: S.eVram === v }, l))),
      h("select", { class: "search", "aria-label": "Dimensione", onchange: (ev) => { S.eSize = ev.target.value; render(); } },
        [["all", "Ogni dimensione"], ["s", "< 1B"], ["m", "1–5B"], ["l", "5–10B"], ["xl", "> 10B"]].map(([v, l]) => h("option", { value: v, selected: S.eSize === v }, l))),
      h("input", { class: "search", type: "search", placeholder: "Cerca…", value: S.eQ, "aria-label": "Cerca embedding",
        oninput: (ev) => { S.eQ = ev.target.value; renderKeepFocus(ev.target); } }),
      latestToggle(hiddenOld)),
    h("div", { class: "tbl-wrap card" }, h("table", { class: "tbl lb emb" },
      h("thead", {}, h("tr", {}, h("th", { class: "num", title: "Posizione in questa vista (con i filtri e l'ordinamento scelti)" }, "#"),
        th("rank", "MTEB", "Posizione nella leaderboard ufficiale MTEB (tutti i modelli valutati)"), th("name", "Modello"), th("mean", "Media", "Media dei task"),
        th("meanType", "Media tipi", "Media per tipo di task"), B.language ? th("lang", langLabel, `Media dei task in ${langLabel.toLowerCase()}`) : null,
        types.map((t) => th("t:" + t, TYPE_IT[t] || t, TYPE_HELP[t])), th("params", "Param."), th("dim", "Dim."), th("maxTokens", "Max token"), th("released", "Uscita"), h("th", {}, gpu))),
      h("tbody", {}, rows.map((r, i) => {
        const m = r.meta;
        const fit = embFits(m);
        return h("tr", {},
          h("td", { class: "num" }, h("b", {}, i + 1)),
          h("td", { class: colCls("rank", "num muted") }, `#${r.rank}`),
          h("td", {}, ext(m.url || `https://huggingface.co/${m.name}`, { class: "lb-name" }, m.name.split("/").pop()),
            h("div", { class: "small" }, h("span", { class: `badge ${m.open ? "open" : "closed"}` }, m.open ? "aperto" : "API"), " ", m.name.includes("/") ? m.name.split("/")[0] : "",
              embNew(m) ? " · nuovo" : "", r.zs != null && r.zs >= 0 && r.zs < 90 ? h("span", { class: "zs", title: "Percentuale di task su cui il modello NON è stato addestrato (zero-shot): sotto 100% i punteggi possono essere gonfiati" }, ` · zero-shot ${r.zs}%`) : "",
              m.newer ? h("span", { class: "newer" }, ` · superato da ${m.newer.split("/").pop()}`) : "")),
          h("td", { class: "wide" }, h("div", { class: "ibar" }, h("i", { style: `width:${Math.max(2, (r.mean || 0) / maxMean * 100)}%` }), h("b", {}, fmtNum(r.mean)))),
          h("td", { class: colCls("meanType") }, fmtNum(r.meanType)),
          B.language ? h("td", { class: colCls("lang") }, fmtNum(r.lang)) : null,
          types.map((t) => h("td", { class: colCls("t:" + t) }, fmtNum(r.types[t]))),
          h("td", { class: "num" }, m.params ? fmtParams(m.params) : "—"), h("td", { class: "num" }, m.dim || "—"),
          h("td", { class: "num" }, m.maxTokens ? fmtCtx(m.maxTokens) : "—"), h("td", { class: "num" }, fmtShort(m.released) || "—"),
          h("td", {}, m.open && fit ? h("span", { class: `lvl ${fit.fits ? "ok" : "no"}` }, `${fit.gb.toFixed(1)} GB`) : h("span", { class: "muted" }, m.open ? "—" : "API")));
      })))),
    h("p", { class: "small source-note" }, `Fonte: leaderboard ufficiale MTEB (${B.name}, ${B.total} modelli valutati; qui i primi 150 più tutti quelli recenti). `,
      "«#» è la posizione in questa vista, «MTEB» la posizione nella leaderboard ufficiale: i salti dipendono dai filtri e dai modelli superati nascosti. ",
      "MTEB misura dataset pubblici: sul tuo caso d'uso (es. il tuo dominio o la tua lingua) l'ordine può cambiare, conviene una prova sui tuoi dati. ",
      "Punteggi ×100. «Zero-shot» indica su quanti task il modello non è stato addestrato: sotto il 100% il punteggio può essere ottimista. ",
      `VRAM = pesi in BF16 + 1,5 GB per l'encoding a batch. Dati al ${fmtShort(E.updated)}.`),
  ];
}

/* ───────────── confronto ───────────── */
const normB =(s) => (s || "").toLowerCase().replace(/\([^)]*\)/g, "").replace(/[^a-z0-9]/g, "");
const num = (s) => { const m = String(s ?? "").replace(",", ".").match(/-?\d+(\.\d+)?/); return m ? +m[0] : null; };

function compareRows(ms) {
  const hw = S.index.hardware;
  const ref = hw.ref_context;
  const recQ = (e) => (e.vram && e.vram.best_quant) || "Q4_K_M";
  const q4 = (e) => localModel(e) ? vramRow(e, bpwOf(recQ(e)) || 4.85, kvKnown(e) ? Math.min(ref, e.arch.max_ctx || ref) : 0) : null;
  const rows = [
    ["Tipo", (e) => KIND[e.kind] || e.kind],
    ["Uscita", (e) => (fmtShort(e.released) || "—") + (isNew(e) ? " · nuovo" : "")],
    ["Versione più recente", (e) => e.newer ? `⚠ ${e.newer.name} (${fmtShort(e.newer.released)})` : "—"],
    ["Parametri", (e) => e.params_b ? fmtParams(e.params_b) + (e.active_b ? ` (${fmtParams(e.active_b)} attivi)` : "") : "—"],
    ["Contesto", (e) => fmtCtx(e.context) || "—", (e) => e.context, "max"],
    ["Licenza", (e) => e.license || (e.kind === "closed" ? "proprietaria" : "—")],
    ["Prezzo API input $/M (OpenRouter)", (e) => e.pricing && e.pricing.in != null ? `$${fmtPrice(e.pricing.in)}` : "—", (e) => e.pricing && e.pricing.in, "min"],
    ["Prezzo API output $/M (OpenRouter)", (e) => e.pricing && e.pricing.out != null ? `$${fmtPrice(e.pricing.out)}` : "—", (e) => e.pricing && e.pricing.out, "min"],
    ["Costo in locale", (e) => localModel(e) ? (runsOnGpu(e) ? "solo hardware ed energia" : e.vram && e.vram.mode === "offload" ? "hardware, ma lento (GPU + RAM)" : "non gira sulla GPU") : "—"],
    ["AA Intelligence", (e) => (e.aa || {}).intelligence != null ? fmtNum(e.aa.intelligence) : "n.v.", (e) => (e.aa || {}).intelligence, "max"],
    ["AA Coding", (e) => (e.aa || {}).coding != null ? fmtNum(e.aa.coding) : "n.v.", (e) => (e.aa || {}).coding, "max"],
    ["AA Agentic", (e) => (e.aa || {}).agentic != null ? fmtNum(e.aa.agentic) : "n.v.", (e) => (e.aa || {}).agentic, "max"],
  ];
  const orKeys = [...new Set(ms.flatMap((e) => Object.keys(e.or_evals || {})))].sort();
  for (const k of orKeys) rows.push([`OpenRouter ${k.replace(/_/g, " ")}`, (e) => (e.or_evals || {})[k] != null ? `${e.or_evals[k]}%` : "—", (e) => (e.or_evals || {})[k], "max"]);
  rows.push(
    [`Gira sulla ${hw.gpu_name}?`, (e) => e.vram ? h("span", { class: `lvl ${e.vram.level}` }, e.vram.text) : (e.kind === "closed" ? "solo API" : "—")],
    ["Formato consigliato", (e) => localModel(e) && e.vram ? (e.vram.quantized ? recQ(e) : "BF16 (nativo)") : "—"],
    [`VRAM nel formato consigliato @${fmtCtx(ref)}`, (e) => q4(e) ? `${q4(e).total.toFixed(1)} GB` : "—", (e) => q4(e) && q4(e).total, "min"],
    ["Contesto max nel formato consigliato", (e) => { const c = localModel(e) && maxCtx(e, bpwOf(recQ(e)) || 4.85); return c ? fmtCtx(c) : c === 0 ? "non entra" : "—"; }, (e) => localModel(e) && maxCtx(e, bpwOf(recQ(e)) || 4.85), "max"],
    ["Velocità (stima)", (e) => { const r = q4(e); return r && r.fits && isTextModel(e) ? `${fmtTok(r.tok)} tok/s` : "—"; }, (e) => { const r = q4(e); return r && r.fits && isTextModel(e) ? r.tok : null; }, "max"],
    ["Checkpoint nativo pronto", (e) => e.vram && e.vram.native_best ? (e.vram.native_ready ? `✓ ${e.vram.native_ready}` : `nessun ${e.vram.native_best}`) : "—"],
    ["Modalità", (e) => e.pipeline || e.modality || "—"],
    ["Knowledge cutoff", (e) => e.knowledge_cutoff || "—"],
  );
  return rows;
}

function scenarios(ms) {
  const gpu = S.index.hardware.gpu_name;
  const aa = (k) => (e) => (e.aa || {})[k];
  const local = ms.filter(runsOnGpu);
  const recCtx = (e) => maxCtx(e, bpwOf(e.vram.best_quant) || 4.85);
  const pick = (list, get, why) => {
    const w = list.filter((e) => { const v = get(e); return typeof v === "number" && !isNaN(v); });
    if (!w.length) return null;
    const best = w.reduce((a, b) => get(b) > get(a) ? b : a);
    const miss = list.filter((e) => !w.includes(e)).map((e) => e.name);
    return { e: best, why: why(best) + (w.length === 1 && list.length > 1 ? " (unico con il dato)" : miss.length ? ` · non valutato: ${miss.join(", ")}` : "") };
  };
  return [
    ["Coding in locale", local.length ? pick(local, aa("coding"), (e) => `AA Coding ${fmtNum(e.aa.coding)}, sulla ${gpu} in ${e.vram.best_quant}`) : null,
      local.length ? "nessun indice Coding per i modelli che girano in locale" : `nessuno gira sulla ${gpu}`],
    ["Massima qualità (API o locale)", pick(ms, aa("intelligence"), (e) => `AA Intelligence ${fmtNum(e.aa.intelligence)}`), "nessun indice Intelligence"],
    ["Agenti e tool use", pick(ms, aa("agentic"), (e) => `AA Agentic ${fmtNum(e.aa.agentic)}`), "nessun indice Agentic"],
    ["API economica", pick(ms.filter((e) => blended(e) > 0 && aa("intelligence")(e) != null), (e) => e.aa.intelligence / blended(e),
      (e) => `${fmtNum(e.aa.intelligence)} punti AA a ~$${fmtPrice(blended(e))} per M token (media 3:1 input/output)`), "servono prezzo API e indice AA"],
    ["Contesto lungo in locale", pick(local.filter(kvKnown), recCtx, (e) => `fino a ~${fmtCtx(recCtx(e))} token in ${e.vram.best_quant}`),
      local.length ? "contesto massimo non stimabile" : `nessuno gira sulla ${gpu}`],
    ["Contesto lungo via API", pick(ms.filter((e) => e.pricing && e.context), (e) => e.context, (e) => `${fmtCtx(e.context)} token`), "nessun modello disponibile via API"],
    ["Velocità in locale", pick(local, (e) => e.vram.tok_s, (e) => `~${e.vram.tok_s} tok/s stimati in ${e.vram.best_quant}`), local.length ? "velocità non stimabile" : `nessuno gira sulla ${gpu}`],
  ];
}

function bestIdx(ms, get, better) {
  if (!get || !better) return new Set();
  const vals = ms.map((e) => { const v = get(e); return typeof v === "number" && !isNaN(v) ? v : null; });
  const nums = vals.filter((v) => v != null);
  if (nums.length < 2) return new Set();
  const target = better === "max" ? Math.max(...nums) : Math.min(...nums);
  return new Set(vals.map((v, i) => v === target ? i : -1).filter((i) => i >= 0));
}

function viewCompare() {
  const ids = cmpGet();
  const ms = ids.map((id) => S.models[id]);
  const all = Object.values(S.models).sort((a, b) => a.name.localeCompare(b.name));
  const label = (e) => `${e.name} — ${e.org || e.kind}`;
  const out = [
    h("div", { class: "section-head" }, h("h2", {}, "Confronto modelli"), h("span", { class: "muted" }, "fino a 4 · aggiungili con «+ confronta» da schede e classifica, o cercali qui")),
    h("div", { class: "filters" },
      h("datalist", { id: "all-models" }, all.map((e) => h("option", { value: label(e) }))),
      h("input", { class: "search wide", list: "all-models", placeholder: "Aggiungi un modello…", "aria-label": "Aggiungi un modello al confronto",
        onchange: (ev) => { const e = all.find((m) => label(m) === ev.target.value); if (e && !cmpGet().includes(e.id)) cmpToggle(e.id); render(); } }),
      ms.length ? h("button", { class: "btn small", type: "button", onclick: () => { cmpSet([]); render(); } }, "svuota") : null),
  ];
  if (!ms.length) {
    const top = Object.values(S.models).filter((e) => e.aa && e.aa.intelligence != null && e.vram && e.vram.level === "ok")
      .sort((a, b) => b.aa.intelligence - a.aa.intelligence).slice(0, 3);
    out.push(h("div", { class: "empty-box" }, h("p", {}, "Nessun modello selezionato."),
      top.length ? h("p", {}, h("button", { class: "btn primary", type: "button", onclick: () => { cmpSet(top.map((e) => e.id)); render(); } },
        `Prova: i 3 migliori che girano sulla ${S.index.hardware.gpu_name}`)) : null));
    return out;
  }
  if (ms.length >= 2) {
    out.push(h("section", { class: "card scen" }, h("h3", {}, "Quale sceglieresti per…"),
      h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" }, h("thead", {}, h("tr", {}, h("th", {}, "Uso"), h("th", {}, "Consigliato"), h("th", {}, "Perché"))),
        h("tbody", {}, scenarios(ms).map(([use, r, miss]) => h("tr", {}, h("th", { scope: "row" }, use),
          r ? h("td", {}, h("a", { href: openModelHash(r.e.id), class: "lb-name" }, r.e.name)) : h("td", { class: "muted" }, "dati insufficienti"),
          h("td", { class: r ? "" : "muted small" }, r ? r.why : miss)))))),
      h("p", { class: "small" }, "Vincitori calcolati solo sui dati disponibili (indici Artificial Analysis, prezzi OpenRouter, stime GPU); per un consiglio sul tuo compito specifico usa il pannello in fondo.")));
  }
  const rows = compareRows(ms);
  const head = h("tr", {}, h("th", { class: "rowlab" }, ""), ms.map((e) => h("th", { class: "mcol" },
    h("div", { class: "mtop" }, kindBadge(e), newBadge(e)),
    h("a", { href: openModelHash(e.id), class: "lb-name" }, e.name), h("div", { class: "small" }, e.org || ""),
    h("button", { class: "x", type: "button", "aria-label": `Togli ${e.name}`, onclick: () => { cmpToggle(e.id); render(); } }, "✕ togli"))));
  const body = rows.map(([lab, show, get, better]) => {
    const best = bestIdx(ms, get, better);
    return h("tr", {}, h("th", { class: "rowlab", scope: "row" }, lab), ms.map((e, i) => h("td", { class: best.has(i) ? "best" : "" }, show(e))));
  });
  // benchmark dichiarati in comune
  const maps = ms.map((e) => { const m = {}; for (const b of ((e.deep || {}).benchmarks || [])) m[normB(b.name)] = b; return m; });
  const keys = [...new Set(maps.flatMap((m) => Object.keys(m)))].filter((k) => maps.filter((m) => m[k]).length >= 2);
  const bench = keys.map((k) => {
    const first = maps.find((m) => m[k])[k];
    const best = bestIdx(ms, (e) => num((maps[ms.indexOf(e)][k] || {}).score), "max");
    return h("tr", {}, h("th", { class: "rowlab", scope: "row" }, first.name), ms.map((e, i) => h("td", { class: best.has(i) ? "best" : "" }, (maps[i][k] || {}).score || "—")));
  });
  out.push(h("div", { class: "tbl-wrap card" }, h("table", { class: "tbl cmp" }, h("thead", {}, head), h("tbody", {}, body,
    bench.length ? [h("tr", { class: "sep" }, h("th", { colspan: ms.length + 1 }, "Benchmark dichiarati dai produttori (in comune)")), bench] : null,
    h("tr", { class: "sep" }, h("th", { colspan: ms.length + 1 }, "Caratteristiche")),
    h("tr", {}, h("th", { class: "rowlab" }, ""), ms.map((e) => h("td", { class: "feat" },
      ((e.deep || {}).features || []).length ? h("ul", {}, e.deep.features.map((f) => h("li", {}, f))) : h("p", { class: "muted" }, (e.deep || {}).tldr || e.description || "—"))))))));
  out.push(h("p", { class: "small source-note" }, "In verde il valore migliore della riga. Gli indici AA e le valutazioni OpenRouter sono misurati in modo uniforme; i benchmark dichiarati possono usare condizioni di test diverse. VRAM e velocità sono stime per la tua GPU."));
  if (ms.length >= 2) out.push(advicePanel(ms));
  return out;
}

/* ───────────── consiglio LLM (dal browser, con la tua chiave OpenRouter) ───────────── */
const getKey = () => store(OR_KEY) || "";
function b64url(bytes) { let s = ""; bytes.forEach((b) => { s += String.fromCharCode(b); }); return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""); }
async function connectOpenRouter() {
  if (!window.crypto || !crypto.subtle) { toast("Serve una pagina https per il collegamento"); return; }
  const verifier = b64url(crypto.getRandomValues(new Uint8Array(32)));
  const challenge = b64url(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier))));
  store("radar-pkce", verifier);
  store("radar-return", location.hash || "#v=compare");
  const cb = location.origin + location.pathname;
  location.href = `https://openrouter.ai/auth?callback_url=${encodeURIComponent(cb)}&code_challenge=${challenge}&code_challenge_method=S256`;
}
async function finishOpenRouter() {
  const code = new URLSearchParams(location.search).get("code");
  if (!code) return;
  const verifier = store("radar-pkce");
  history.replaceState(null, "", location.pathname + (store("radar-return") || "#v=compare"));
  try {
    const r = await fetch("https://openrouter.ai/api/v1/auth/keys", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, code_verifier: verifier, code_challenge_method: "S256" }) });
    const d = await r.json();
    if (d.key) { store(OR_KEY, d.key); toast("OpenRouter collegato ✓"); } else toast("Collegamento non riuscito");
  } catch { toast("Collegamento non riuscito"); }
  store("radar-pkce", null);
}

function compactModel(e) {
  const r = localModel(e) ? vramRow(e, 4.85, kvKnown(e) ? Math.min(S.index.hardware.ref_context, e.arch.max_ctx || 1e9) : 0) : null;
  const d = e.deep || {};
  return {
    nome: e.name, produttore: e.org, tipo: KIND[e.kind] || e.kind, uscita: e.released, nuovo: isNew(e),
    versione_piu_recente: e.newer ? `${e.newer.name} (${e.newer.released})` : null,
    parametri_B: e.params_b, attivi_B: e.active_b, contesto: e.context, licenza: e.license, modalita: e.pipeline || e.modality,
    prezzo_usd_per_M: e.pricing ? { input: e.pricing.in, output: e.pricing.out } : null,
    artificial_analysis: e.aa ? { intelligence: e.aa.intelligence, coding: e.aa.coding, agentic: e.aa.agentic } : null,
    valutazioni_openrouter_pct: e.or_evals || null,
    gpu_verdetto: e.vram ? e.vram.text : (e.kind === "closed" ? "solo API" : null),
    q4_vram_gb: r ? +r.total.toFixed(1) : null, q4_contesto_max: localModel(e) ? maxCtx(e, 4.85) : null,
    q4_tok_s: r && r.fits && isTextModel(e) ? Math.round(r.tok) : null,
    opzioni_gpu: e.vram && e.vram.options ? e.vram.options.filter((o) => o.fits).map((o) => `${o.q}${o.native ? " (nativo)" : ""}: ${o.comfy ? "entra" : "al limite"}, ctx max ${o.max_ctx || "?"}, ~${o.tok_s || "?"} tok/s`) : null,
    checkpoint_nativo_pronto: e.vram ? e.vram.native_ready || null : null,
    benchmark_dichiarati: (d.benchmarks || []).slice(0, 12).map((b) => `${b.name}: ${b.score}`),
    caratteristiche: d.features || [], sintesi: d.tldr || e.description || null,
  };
}

async function askAdvice(ms, task) {
  const hw = S.index.hardware;
  const sys = `Sei un consulente tecnico esperto di modelli AI. Profilo dell'utente:\n${S.index.site.profile}\n` +
    `Hardware: ${hw.gpu_name} con ${hw.vram_gb} GB di VRAM, ${hw.system_ram_gb} GB di RAM.\n` +
    "Basati SOLO sui dati forniti; se mancano dati per decidere dillo chiaramente. Tieni conto di costo, qualità, " +
    "esecuzione locale vs API e licenza in base al compito. Scrivi in italiano. Rispondi SOLO con JSON valido.";
  const user = `Compito dell'utente: ${task}\n\nModelli da confrontare (dati raccolti da AI Radar):\n${JSON.stringify(ms.map(compactModel))}\n\n` +
    'Restituisci: {"vincitore": "nome esatto del modello consigliato", "sintesi": "2-3 frasi sul perché", ' +
    '"classifica": [{"modello": "nome", "voto": 0-10, "pro": ["..."], "contro": ["..."]}], ' +
    '"come_usarlo": "consigli pratici: quantizzazione e contesto sulla GPU, oppure via API con costo stimato", "attenzione": "limiti dei dati o dubbi, oppure stringa vuota"}';
  const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${getKey()}`, "Content-Type": "application/json", "X-Title": "AI Radar (personal)" },
    body: JSON.stringify({ model: S.index.site.advice_model, temperature: 0.3, max_tokens: 2000, usage: { include: true },
      response_format: { type: "json_object" }, messages: [{ role: "system", content: sys }, { role: "user", content: user }] }),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error((d.error && d.error.message) || `HTTP ${r.status}`);
  const txt = (((d.choices || [])[0] || {}).message || {}).content || "";
  const m = txt.replace(/```(json)?/g, "").match(/\{[\s\S]*\}/);
  if (!m) throw new Error("risposta non valida");
  return { res: JSON.parse(m[0]), cost: d.usage && d.usage.cost };
}

function renderAdvice(res, cost) {
  const cls = res.classifica || [];
  return [
    h("div", { class: "winner" }, h("div", { class: "small" }, "Consigliato"), h("h3", {}, res.vincitore || "—"), res.sintesi ? h("p", {}, res.sintesi) : null),
    cls.length ? h("div", { class: "ranking" }, cls.map((c) => h("div", { class: "rank-row" },
      h("div", { class: "rank-head" }, h("b", {}, c.modello), h("div", { class: "ibar" }, h("i", { style: `width:${Math.max(3, (+c.voto || 0) * 10)}%` }), h("b", {}, `${c.voto}/10`))),
      h("div", { class: "procon" },
        (c.pro || []).length ? h("ul", { class: "pro" }, c.pro.map((x) => h("li", {}, x))) : null,
        (c.contro || []).length ? h("ul", { class: "contro" }, c.contro.map((x) => h("li", {}, x))) : null)))) : null,
    res.come_usarlo ? h("div", { class: "md-sec" }, h("h3", {}, "Come usarlo"), h("p", {}, res.come_usarlo)) : null,
    res.attenzione ? h("p", { class: "note" }, "⚠ ", res.attenzione) : null,
    h("p", { class: "small" }, `Risposta di ${S.index.site.advice_model}`, cost ? ` · costo $${(+cost).toFixed(4)}` : ""),
  ];
}

function advicePanel(ms) {
  const out = h("div", { class: "advice-out" });
  const ta = h("textarea", { class: "task", rows: 3, "aria-label": "Descrivi il compito",
    placeholder: "Per cosa ti serve? Es. «RAG su documenti tecnici in italiano, in locale sulla 5090» oppure «agente di coding via API con budget di 20 $/mese»" });
  ta.value = store("radar-task") || "";
  const btn = h("button", { class: "btn primary", type: "button" }, "Chiedi consiglio");
  btn.addEventListener("click", async () => {
    const task = ta.value.trim();
    if (!task) { ta.focus(); toast("Scrivi prima il compito"); return; }
    store("radar-task", task);
    btn.disabled = true;
    btn.textContent = "Sto valutando…";
    out.replaceChildren(h("p", { class: "muted" }, "Il modello sta confrontando i dati…"));
    try {
      const { res, cost } = await askAdvice(ms, task);
      out.replaceChildren(...renderAdvice(res, cost));
    } catch (err) {
      out.replaceChildren(h("p", { class: "note" }, `Errore: ${err.message}. `, /401|key|auth/i.test(err.message) ? "La chiave non è valida: ricollega OpenRouter dalle impostazioni (⚙)." : ""));
    }
    btn.disabled = false;
    btn.textContent = "Chiedi consiglio";
  });
  const connected = !!getKey();
  return h("section", { class: "advice card" },
    h("h3", {}, "Quale scegliere per il tuo compito?"),
    h("p", { class: "small" }, "Un LLM legge tutti i dati del confronto (indici, benchmark, prezzi, verdetto GPU) e ti consiglia il modello migliore per quello che devi fare."),
    connected ? [ta, h("div", { class: "row" }, btn, h("span", { class: "small" }, `${S.index.site.advice_model} · ~0,002 $ a domanda`)), out]
      : h("div", { class: "connect" }, h("p", {}, "Per usarlo collega il tuo account OpenRouter: la chiave resta solo in questo browser."),
        h("button", { class: "btn primary", type: "button", onclick: connectOpenRouter }, "Collega OpenRouter"),
        h("button", { class: "btn", type: "button", onclick: openSettings }, "Incolla una chiave")));
}

/* ───────────── dettaglio modello ───────────── */
function vramCalc(e) {
  const hw = S.index.hardware;
  const a = e.arch;
  const known = kvKnown(e);
  const maxPos = (a && a.max_ctx) || e.context || 131072;
  const steps = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576].filter((c) => c <= maxPos);
  if (!steps.length || steps[steps.length - 1] < maxPos) steps.push(maxPos);
  let ci = Math.max(0, steps.findIndex((c) => c >= Math.min(hw.ref_context, maxPos)));
  let kvi = 0;
  const out = h("output", {});
  const tbody = h("tbody", {});
  const parts = h("p", { class: "small breakdown" });
  const margin = hw.margin_gb ?? 1.5;
  const text = isTextModel(e);
  const update = () => {
    const ctx = steps[ci];
    out.textContent = fmtCtx(ctx);
    const kvGb = known ? kvBytes(a, ctx, KV_TYPES[kvi][1]) / GiB : 0;
    parts.replaceChildren(known ? `A ${fmtCtx(ctx)} di contesto la KV cache (${KV_TYPES[kvi][0]}) occupa ${kvGb.toFixed(1)} GB, uguale per ogni quantizzazione; ` : "",
      "overhead 0,75 GB + 2% dei pesi. Passa il mouse sul totale per la scomposizione.");
    const recDone = {};
    const groups = [["gguf", QUANTS], ["native", nativeDefs()]];
    tbody.replaceChildren(...groups.flatMap(([g, defs]) => (defs.length && g === "native" ? [h("tr", { class: "sep" },
      h("th", { colspan: 6 }, `Formati nativi della ${hw.gpu_name} (accelerazione hardware: vLLM, SGLang, TensorRT-LLM; NVFP4 anche in GGUF)`))] : [])
      .concat(defs.map(([q, bpw]) => {
      const r = vramRow(e, bpw, known ? ctx : 0, KV_TYPES[kvi][1]);
      const comfy = r.total <= hw.vram_gb - hw.reserved_gb - margin;
      const off = !r.fits && g === "gguf" && q !== "BF16" ? offloadRow(e, bpw, known ? ctx : 0, KV_TYPES[kvi][1]) : null;
      const rec = comfy && !recDone[g];
      if (rec) recDone[g] = true;
      const pct = Math.min(100, r.total / hw.vram_gb * 100);
      const split = `pesi ${r.w.toFixed(1)} GB + KV cache ${kvGb.toFixed(1)} GB + overhead ${(r.total - r.w - kvGb).toFixed(1)} GB`;
      return h("tr", { class: rec ? "rec" : "" },
        h("td", {}, h("b", {}, q)), h("td", { class: "num" }, `${r.w.toFixed(1)} GB`), h("td", { class: "num", title: split }, `${r.total.toFixed(1)} GB`),
        h("td", {}, h("div", { class: "bar" }, h("i", { class: comfy ? "ok" : r.fits ? "tight" : off ? "off" : "no", style: `width:${pct}%` }))),
        h("td", { class: comfy ? "fit-ok" : r.fits ? "fit-tight" : off ? "fit-off" : "fit-no",
          title: off ? `~${off.vram.toFixed(1)} GB in VRAM + ~${off.ram.toFixed(1)} GB in RAM` : null },
          rec ? (q === "BF16" ? "★ nativo" : "★ consigliata") : comfy ? "entra" : r.fits ? "al limite" : off ? `GPU + RAM (${off.ram.toFixed(0)} GB in RAM)` : "no"),
        h("td", { class: "num" }, r.fits && text ? `${fmtTok(r.tok)} tok/s` : off && text ? h("span", { class: "muted", title: "con offload in RAM" }, `${fmtTok(off.tok)} tok/s`) : "—"));
    }))));
  };
  const slider = h("input", { type: "range", min: 0, max: steps.length - 1, value: ci, "aria-label": "Contesto", oninput: (ev) => { ci = +ev.target.value; update(); } });
  const kvSel = h("select", { class: "search", "aria-label": "Tipo KV cache", onchange: (ev) => { kvi = +ev.target.value; update(); } },
    KV_TYPES.map(([n], i) => h("option", { value: i }, `KV cache ${n}`)));
  update();
  return h("div", {},
    h("div", { class: "calc-ctl" },
      known ? h("label", {}, "Contesto ", slider, " ", out) : h("span", { class: "small" }, "Architettura non disponibile: stima dei soli pesi."),
      known ? kvSel : null),
    h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" },
      h("thead", {}, h("tr", {}, ["Quant", "Pesi", "Totale", `su ${hw.vram_gb} GB`, "", text ? "Velocità stimata" : ""].map((t) => h("th", {}, t)))), tbody)),
    parts,
    h("details", { class: "method" }, h("summary", {}, "Come è calcolato e quanto è affidabile"),
      h("ul", {},
        h("li", {}, h("b", {}, "Pesi"), " = parametri × bit per peso ÷ 8 (bit effettivi dei GGUF di llama.cpp; FP8 e NVFP4 includono le scale)."),
        h("li", {}, h("b", {}, "KV cache"), " = per ogni layer di attenzione 2 × kv_heads × head_dim × byte × contesto, letti dal config.json. ",
          "Sliding window: limitata alla finestra; MLA: cache compressa; layer lineari/SSM: stato costante."),
        h("li", {}, h("b", {}, "Overhead"), " = 0,75 GB + 2% dei pesi (contesto CUDA e buffer). Il motore reale può usarne di più (vLLM prealloca la memoria)."),
        h("li", {}, h("b", {}, "Velocità"), ` = banda di memoria (${hw.bandwidth_gbs} GB/s) × efficienza ÷ byte letti per token (parametri attivi + KV cache). `,
          "È una stima teorica per una sola richiesta (batch 1) a contesto breve, tipo llama.cpp: non è una misura. ",
          "Con contesti lunghi scende; vLLM con più richieste in parallelo ha un throughput totale più alto."),
        h("li", {}, h("b", {}, "GPU + RAM"), ` = ciò che non entra in VRAM va in RAM (${hw.system_ram_gb} GB, di cui ${hw.ram_reserved_gb ?? 8} lasciati al sistema; banda ~${hw.ram_bandwidth_gbs || 80} GB/s): `,
          "la velocità dipende soprattutto dalla RAM. Con i MoE conviene tenere in RAM gli esperti (llama.cpp --n-cpu-moe)."),
        h("li", {}, h("b", {}, "Precisione"), ": circa ±10% sulla memoria e ±25% sulla velocità. Nessun valore qui è misurato su hardware reale."))),
    h("p", { class: "small" }, `Stima per ${hw.gpu_name} (${hw.vram_gb} GB, ${hw.reserved_gb} GB riservati). `,
      `«Consigliata» = la quantizzazione di qualità più alta che lascia almeno ${margin} GB liberi; «al limite» = entra ma senza margine. `,
      e.active_b ? "MoE: in VRAM vanno tutti i parametri, la velocità dipende da quelli attivi. " : "",
      a && a.linear_layers ? `Attenzione ibrida: solo ${a.full_layers + (a.sliding_layers || 0)} layer su ${a.layers} hanno KV cache che cresce col contesto. ` : "",
      a && a.mla_dim ? "MLA: KV cache compressa. " : ""));
}

/* «Quando sceglierlo»: punti di forza e limiti ricavati dai dati (indici AA, prezzi, contesto, licenza, verdetto GPU) */
function rankAmong(list, get, e) {
  const v = get(e);
  const vals = list.map(get).filter((x) => x != null);
  if (v == null || vals.length < 8) return null;
  const pos = vals.filter((x) => x > v).length + 1;
  return { pos, n: vals.length, pct: pos / vals.length };
}
const blended = (m) => m.pricing && m.pricing.in != null && m.pricing.out != null ? (3 * m.pricing.in + m.pricing.out) / 4 : null;
const runsOnGpu = (m) => !!(m.vram && m.vram.level === "ok" && (m.vram.mode === "bf16" || m.vram.mode === "quant" || !m.vram.mode));
function alternatives(e) {
  const all = Object.values(S.models).filter((m) => m.id !== e.id && !m.newer);
  const ai = (m) => (m.aa || {}).intelligence;
  if (runsOnGpu(e)) {
    let alt = all.filter((m) => runsOnGpu(m) && ai(m) != null);
    if (ai(e) != null) alt = alt.filter((m) => ai(m) >= ai(e) - 3);
    else if (e.params_b) alt = alt.filter((m) => m.params_b && m.params_b >= e.params_b / 2 && m.params_b <= e.params_b * 2);
    return { why: `girano sulla ${S.index.hardware.gpu_name}` + (ai(e) != null ? " con intelligenza simile o superiore" : " con taglia simile"),
      list: alt.sort((a, b) => ai(b) - ai(a)).slice(0, 3) };
  }
  if (ai(e) != null && blended(e) != null) {
    const alt = all.filter((m) => ai(m) != null && blended(m) != null && Math.abs(ai(m) - ai(e)) <= 5 && m.kind !== "announced");
    return { why: "API con intelligenza simile, dalla più economica", list: alt.sort((a, b) => blended(a) - blended(b)).slice(0, 3) };
  }
  if (ai(e) != null) {
    const alt = all.filter((m) => ai(m) != null && Math.abs(ai(m) - ai(e)) <= 4);
    return { why: "intelligenza simile", list: alt.sort((a, b) => ai(b) - ai(a)).slice(0, 3) };
  }
  return { why: "", list: [] };
}
function whenToChoose(e) {
  const all = Object.values(S.models).filter((m) => !m.newer);
  const gpu = S.index.hardware.gpu_name;
  const pros = [], cons = [];
  const dims = [["intelligence", "qualità generale"], ["coding", "coding"], ["agentic", "agenti e tool use"]];
  for (const [k, lab] of dims) {
    const r = rankAmong(all, (m) => (m.aa || {})[k], e);
    if (r && r.pct <= 0.15) pros.push(`Tra i migliori per ${lab}: #${r.pos} su ${r.n} modelli valutati da Artificial Analysis`);
    else if (r && r.pct >= 0.75) cons.push(`Indietro per ${lab}: #${r.pos} su ${r.n}`);
  }
  if (runsOnGpu(e)) {
    const loc = all.filter(runsOnGpu);
    for (const [k, lab] of dims.slice(0, 2)) {
      const r = rankAmong(loc, (m) => (m.aa || {})[k], e);
      if (r && r.pos <= 3) pros.push(`#${r.pos} per ${lab} tra i ${r.n} modelli valutati che girano sulla ${gpu}`);
    }
  }
  const ai = (e.aa || {}).intelligence;
  if (blended(e) != null && ai != null) {
    const peers = all.filter((m) => blended(m) != null && (m.aa || {}).intelligence != null && Math.abs(m.aa.intelligence - ai) <= 4);
    if (peers.length >= 5) {
      const share = peers.filter((m) => blended(m) < blended(e)).length / peers.length;
      if (share <= 0.25) pros.push("API economica per il suo livello di intelligenza");
      else if (share >= 0.75) cons.push("API cara rispetto a modelli di intelligenza simile");
    }
  }
  if (e.context >= 500000) pros.push(`Contesto molto lungo (${fmtCtx(e.context)})`);
  else if (e.context >= 200000) pros.push(`Contesto lungo (${fmtCtx(e.context)})`);
  if (/image|vision|any-to-any|multimodal|video/i.test(`${e.pipeline || ""} ${e.modality || ""}`)) pros.push("Multimodale: accetta anche immagini");
  if (/apache|\bmit\b|bsd/i.test(e.license || "")) pros.push(`Licenza permissiva (${e.license})`);
  else if (e.license && /other|research|non-?commercial|\bnc\b/i.test(e.license)) cons.push(`Licenza da leggere prima dell'uso (${e.license})`);
  const v = e.vram;
  if (v && v.mode) {
    if (v.mode === "bf16") pros.push(`Gira sulla ${gpu} senza quantizzare`);
    else if (v.mode === "quant" && v.level === "ok") pros.push(`Gira sulla ${gpu} quantizzato (${v.best_quant})`);
    else if (v.mode === "quant") cons.push(`Sulla ${gpu} solo con quantizzazione aggressiva`);
    else if (v.mode === "offload") cons.push(`Sulla ${gpu} solo con offload in RAM${v.offload && v.offload.tok_s ? ` (~${v.offload.tok_s} tok/s stimati)` : ""}`);
    else cons.push(`Troppo grande per la ${gpu}, anche con offload`);
  } else if (e.kind === "closed") cons.push("Solo API: niente esecuzione locale");
  if (e.gated) cons.push("Pesi gated: bisogna accettare le condizioni su Hugging Face");
  if (e.newer) cons.push(`Superato da ${e.newer.name}`);
  const noData = !e.aa && !((e.deep || {}).benchmarks || []).length;
  const alt = alternatives(e);
  if (!pros.length && !cons.length && !alt.list.length) return null;
  return h("div", { class: "md-sec choose" }, h("h3", {}, "Quando sceglierlo"),
    h("div", { class: "procon" },
      pros.length ? h("ul", { class: "pro" }, pros.map((x) => h("li", {}, x))) : null,
      cons.length ? h("ul", { class: "contro" }, cons.map((x) => h("li", {}, x))) : null),
    noData ? h("p", { class: "small" }, "Nessuna valutazione indipendente: punti di forza ricavati solo da scheda e verdetto GPU.") : null,
    alt.list.length ? h("div", { class: "alts" }, h("span", { class: "small" }, `Alternative (${alt.why}): `),
      alt.list.map((m, i) => [i ? " · " : "", h("a", { href: openModelHash(m.id) }, m.name), (m.aa || {}).intelligence != null ? h("span", { class: "small" }, ` AA ${fmtNum(m.aa.intelligence)}`) : ""]),
      " ", h("button", { class: "btn small", type: "button", onclick: () => { cmpSet([e.id, ...alt.list.map((m) => m.id)]); modal.close(); location.hash = "#v=compare"; } }, "Confrontali")) : null,
    h("p", { class: "small" }, "Calcolato dai dati raccolti (indici AA, prezzi, contesto, licenza, stima GPU), non è un giudizio dell'LLM."));
}

/* comandi suggeriti per avviarlo in locale */
function runCommands(e) {
  const v = e.vram;
  if (!v || !e.hf_id || !isTextModel(e) || v.mode === "no" || !v.mode) return null;
  const hw = S.index.hardware;
  const ctx = Math.min(hw.ref_context, (e.arch && e.arch.max_ctx) || e.context || hw.ref_context);
  const L = (e.arch && e.arch.layers) || null;
  const cmds = [];
  const gguf = (e.quants || []).find((q) => q.format === "GGUF" && !q.nvfp4) || (e.quants || []).find((q) => q.format === "GGUF");
  const q = v.mode === "offload" ? v.offload && v.offload.q : v.gguf_best || (v.mode === "bf16" ? "Q8_0" : null);
  if (gguf && q) {
    let c = `llama-server -hf ${gguf.id}:${q} -c ${ctx} -ngl 99`;
    let note = "";
    if (v.mode === "offload" && v.offload) {
      const frac = v.offload.ram_gb / (v.offload.ram_gb + v.offload.vram_gb);
      if (v.is_moe) { c += ` --n-cpu-moe ${L ? Math.min(L, Math.ceil(L * frac * 1.15)) : "N"}`; note = "--n-cpu-moe N = esperti dei primi N layer in RAM: aumentalo se esaurisci la VRAM, abbassalo se avanza."; }
      else { c = c.replace("-ngl 99", `-ngl ${L ? Math.max(1, Math.floor(L * (1 - frac))) : "N"}`); note = "-ngl N = layer in GPU, il resto gira in RAM: regolalo finché la VRAM non è quasi piena."; }
    }
    cmds.push(["llama.cpp (GGUF)", c, note]);
    cmds.push(["Ollama", `ollama run hf.co/${gguf.id}:${q}`, ""]);
  }
  if (v.mode === "quant" && v.native_ready) cmds.push([`vLLM (${v.native_best} nativo)`, `vllm serve ${v.native_ready} --max-model-len ${ctx} --gpu-memory-utilization 0.92`, ""]);
  if (v.mode === "bf16") cmds.push(["vLLM (BF16)", `vllm serve ${e.hf_id} --max-model-len ${ctx} --gpu-memory-utilization 0.92`, ""]);
  if (!cmds.length) return null;
  const copy = (txt) => { try { navigator.clipboard.writeText(txt); toast("Comando copiato"); } catch { toast("Copia non disponibile"); } };
  return h("div", { class: "md-sec" }, h("h3", {}, "Avvio rapido"),
    cmds.map(([t, c, note]) => h("div", { class: "cmd" }, h("div", { class: "cmd-head" }, h("b", {}, t),
      h("button", { class: "btn small", type: "button", onclick: () => copy(c) }, "copia")), h("pre", {}, h("code", {}, c)), note ? h("p", { class: "small" }, note) : null)),
    h("p", { class: "small" }, "Comandi suggeriti in base alla stima, non testati. ",
      gguf && q ? `Se nel repo GGUF manca la variante ${q}, scegline un'altra dalla pagina Hugging Face. ` : "",
      e.gated ? "Repo gated: accetta le condizioni su HF e usa un token (HF_TOKEN). " : "",
      "Controlla i flag sulla documentazione di llama.cpp / vLLM."));
}

/* da dove viene ogni dato della scheda */
const PARAM_SRC = { safetensors: "verificati dai metadati dei pesi su Hugging Face", nome: "dedotti dal nome del repository", "stima da config": "stimati dalla configurazione del modello" };
function dataQuality(e) {
  const d = e.deep || {};
  const rows = [];
  const av = availability(e);
  rows.push(["Disponibilità", `${av.label}` + (e.hf_checked ? ` · repo verificato su Hugging Face il ${fmtShort(e.hf_checked)}` : e.hf_id ? " · da Hugging Face" : e.or_id ? " · da OpenRouter" : "")]);
  if (e.params_b) rows.push(["Parametri", PARAM_SRC[e.params_source] || (e.kind === "closed" ? "dichiarati" : "fonte non indicata")]);
  if (e.active_b) rows.push(["Parametri attivi", e.active_source === "dichiarato" ? "dichiarati dal produttore (config.json non leggibile)" : e.arch ? "calcolati dall'architettura" : "dal nome del modello"]);
  if (localModel(e)) rows.push(["Architettura", e.arch ? "dal config.json su Hugging Face (KV cache, contesto, esperti)" : "config.json non disponibile"]);
  if (e.quants_checked) rows.push(["Quantizzazioni", `cercate su Hugging Face il ${fmtShort(e.quants_checked)}`]);
  if (e.aa) rows.push(["Indici AA", "misurati da Artificial Analysis (terza parte, stessa metodologia per tutti)"]);
  if ((d.benchmarks || []).length) rows.push(["Benchmark", `dichiarati dal produttore${d.source ? ` (${host(d.source)})` : ""}: condizioni di test non uniformi`]);
  if (d.hw_claims) rows.push(["Requisiti dichiarati", "dal testo del produttore, non verificati"]);
  if (e.vram) rows.push(["VRAM e velocità", "stime di AI Radar dall'architettura, non misure"]);
  if (d.by) rows.push(["Sintesi e caratteristiche", `estratte da ${d.by} il ${fmtShort(d.date)}`]);
  return h("details", { class: "md-sec quality" }, h("summary", {}, h("h3", { class: "inline" }, "Qualità dei dati"),
      (e.checks || []).length ? h("span", { class: "badge announced" }, plural(e.checks.length, "avviso", "avvisi")) : h("span", { class: "small" }, " da dove viene ogni informazione")),
    (e.checks || []).length ? h("ul", { class: "checks" }, e.checks.map((c) => h("li", {}, "⚠ ", c))) : null,
    h("div", { class: "kv" }, rows.map(([k, x]) => h("div", {}, h("span", {}, k + ": "), x))),
    e.updated ? h("p", { class: "small" }, `Scheda aggiornata il ${fmtShort(e.updated)}.`) : null);
}

function renderModel(e) {
  const d = e.deep || {};
  const secs = [];
  secs.push(h("div", { class: "md-head" },
    h("div", { class: "mtop" }, kindBadge(e), newBadge(e), h("span", { class: "muted" }, [e.org, e.released ? `uscito ${fmtShort(e.released)}` : null, e.first_seen ? `visto ${fmtShort(e.first_seen)}` : null].filter(Boolean).join(" · ")), h("span", { class: "spacer" }), cmpBtn(e)),
    h("h2", {}, e.name), modelChips(e)));
  const v = verdict(e);
  if (v) secs.push(h("div", { class: "md-sec" }, v));
  if (d.hw_claims) secs.push(h("div", { class: "md-sec claim" }, h("b", {}, "Dichiarato dal produttore: "), d.hw_claims,
    d.source ? h("span", { class: "small" }, " — ", ext(d.source, host(d.source))) : null,
    h("div", { class: "small" }, "Requisiti riportati nella fonte ufficiale (spesso con un motore di inferenza specifico): possono differire dalla stima generica di AI Radar.")));
  const nn = newerNote(e);
  if (nn) secs.push(nn);
  if (e.note) secs.push(h("p", { class: "note md-sec" }, e.note));
  if (d.tldr || e.description) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "In breve"), h("p", {}, d.tldr || e.description)));
  const wc = whenToChoose(e);
  if (wc) secs.push(wc);
  if (d.features && d.features.length) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Caratteristiche"), h("ul", {}, d.features.map((f) => h("li", {}, f)))));
  const facts = [["Parametri dichiarati", d.params], ["Contesto dichiarato", d.context], ["Disponibilità", d.availability],
    ["Knowledge cutoff", e.knowledge_cutoff], ["Accesso ai pesi", e.gated ? "gated: accetta le condizioni sulla pagina Hugging Face" : null]].filter(([, x]) => x);
  if (facts.length) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Dati"), h("div", { class: "kv" }, facts.map(([k, x]) => h("div", {}, h("span", {}, k + ": "), x)))));
  if (e.aa || e.or_evals) {
    const meta = S.index.benchmarks_meta || {};
    secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Valutazioni indipendenti"),
      h("div", { class: "kv" },
        e.aa ? [["Intelligence Index", e.aa.intelligence], ["Coding Index", e.aa.coding], ["Agentic Index", e.aa.agentic]].filter(([, x]) => x != null)
          .map(([k, x]) => h("div", {}, h("span", {}, `AA ${k}: `), h("b", {}, fmtNum(x)))) : null,
        Object.entries(e.or_evals || {}).map(([k, x]) => h("div", {}, h("span", {}, `OpenRouter ${k.replace(/_/g, " ")}: `), h("b", {}, `${x}%`)))),
      h("p", { class: "small" }, e.aa && e.aa.variant ? `Variante misurata: ${e.aa.variant}. ` : "", meta.citation || "Fonte: Artificial Analysis via OpenRouter.",
        " ", h("a", { href: "#v=leaderboard" }, "Vedi la classifica →"))));
  }
  if (d.benchmarks && d.benchmarks.length) {
    secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Benchmark dichiarati"),
      h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" },
        h("thead", {}, h("tr", {}, h("th", {}, "Benchmark"), h("th", {}, e.name), h("th", {}, "Confronto"))),
        h("tbody", {}, d.benchmarks.map((b) => h("tr", {}, h("td", {}, b.name), h("td", { class: "num" }, h("b", {}, b.score)),
          h("td", { class: "vs" }, (b.vs || []).map((x) => `${x.model} ${x.score}`).join(" · "))))))),
      d.source ? h("p", { class: "small" }, "Fonte: ", ext(d.source, host(d.source)), d.by ? ` · estratto da ${d.by}` : "") : null));
  }
  if (localModel(e)) secs.push(h("div", { class: "md-sec" }, h("h3", {}, `Gira sulla tua ${S.index.hardware.gpu_name}?`), vramCalc(e)));
  if (e.quants && e.quants.length) {
    secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Quantizzazioni pronte"),
      h("div", { class: "quants" }, e.quants.map((q) => ext(`https://huggingface.co/${q.id}`, { class: "mlink q-" + q.format.toLowerCase() },
        h("b", {}, q.format + (q.nvfp4 && q.format === "GGUF" ? " NVFP4" : "")), ` ${q.id}`, q.official ? " · ufficiale" : "")))));
  }
  const rc = runCommands(e);
  if (rc) secs.push(rc);
  if (d.local_tips) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Consigli per l'uso locale (dalla fonte)"), h("p", {}, d.local_tips)));
  if (e.mentions && e.mentions.length) {
    secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Ne parlano"),
      h("ul", {}, e.mentions.map((m) => h("li", {}, ext(m.url, m.title), h("span", { class: "small" }, ` — ${m.source}, ${fmtShort(m.date)}`))))));
  }
  const L = e.links || {};
  const links = [L.hf && ["Hugging Face", L.hf], L.openrouter && ["OpenRouter", L.openrouter], L.announcement && ["Annuncio", L.announcement]].filter(Boolean);
  if (links.length) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Link"), h("div", { class: "chips" }, links.map(([n, u]) => ext(u, { class: "mlink" }, n)))));
  secs.push(dataQuality(e));
  return secs;
}

const modal = document.getElementById("modal");
function openModel(id) {
  const e = S.models[id];
  if (!e) return;
  document.getElementById("modal-body").replaceChildren(...renderModel(e));
  if (!modal.open) modal.showModal();
  modal.scrollTop = 0;
}
function closeModalHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.has("m")) { p.delete("m"); history.replaceState(null, "", "#" + p.toString()); }
}
modal.addEventListener("close", closeModalHash);
modal.addEventListener("click", (ev) => { if (ev.target === modal) modal.close(); });
modal.querySelector(".modal-close").addEventListener("click", () => modal.close());

/* ───────────── impostazioni ───────────── */
const sdlg = document.getElementById("settings-dlg");
function openSettings() {
  const key = getKey();
  const inp = h("input", { class: "search wide", type: "password", placeholder: "sk-or-v1-…", "aria-label": "Chiave OpenRouter", autocomplete: "off" });
  document.getElementById("settings-body").replaceChildren(
    h("h2", {}, "Impostazioni"),
    h("div", { class: "md-sec" }, h("h3", {}, "OpenRouter (facoltativo)"),
      h("p", {}, key ? `Collegato · chiave …${key.slice(-4)}` : "Non collegato: tutto il sito funziona lo stesso."),
      h("ul", { class: "small" },
        h("li", {}, h("b", {}, "Serve solo per: "), "il consiglio «Quale scegliere per il tuo compito?» nel Confronto e le spiegazioni dei paper che non hanno ancora un'overview."),
        h("li", {}, h("b", {}, "Non serve per: "), "rassegna, schede, classifiche, embedding e overview già pronte (le genera ogni mattina il workflow con la chiave del repository)."),
        h("li", {}, h("b", {}, "Costo: "), `addebitato sul tuo account OpenRouter, ~0,001-0,002 $ a domanda (${S.index.site.advice_model}).`)),
      h("div", { class: "row" },
        h("button", { class: "btn primary", type: "button", onclick: connectOpenRouter }, key ? "Ricollega con OpenRouter" : "Collega con OpenRouter"),
        key ? h("button", { class: "btn", type: "button", onclick: () => { store(OR_KEY, null); toast("Chiave rimossa"); openSettings(); render(); } }, "Scollega") : null),
      h("p", { class: "small" }, "Oppure incolla una chiave:"),
      h("div", { class: "row" }, inp, h("button", { class: "btn", type: "button", onclick: () => {
        const k = inp.value.trim();
        if (!/^sk-or-/.test(k)) { toast("Non sembra una chiave OpenRouter"); return; }
        store(OR_KEY, k); toast("Chiave salvata"); openSettings(); render();
      } }, "Salva")),
      h("p", { class: "small" }, "La chiave resta solo in questo browser e viene inviata solo a openrouter.ai. Consiglio: usa una chiave dedicata con un limite di spesa (es. 2 $), impostabile su openrouter.ai/keys.")),
    h("div", { class: "md-sec" }, h("h3", {}, "Hardware dei verdetti"),
      h("p", {}, `${S.index.hardware.gpu_name}, ${S.index.hardware.vram_gb} GB VRAM, ${S.index.hardware.bandwidth_gbs} GB/s; ${S.index.hardware.system_ram_gb} GB RAM `,
        `(${S.index.hardware.ram_reserved_gb ?? 8} GB lasciati al sistema, ~${S.index.hardware.ram_bandwidth_gbs || 80} GB/s per l'offload). Si cambia in config.toml.`)));
  if (!sdlg.open) sdlg.showModal();
}
sdlg.addEventListener("click", (ev) => { if (ev.target === sdlg) sdlg.close(); });
sdlg.querySelector(".modal-close").addEventListener("click", () => sdlg.close());
document.getElementById("settings").addEventListener("click", openSettings);

/* ───────────── routing e render ───────────── */
const app = document.getElementById("app");
function currentView() { return new URLSearchParams(location.hash.slice(1)).get("v") || "day"; }
function render() {
  const v = currentView();
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.view === v));
  const views = { models: viewModels, leaderboard: viewLeaderboard, papers: viewPapers, embeddings: viewEmbeddings, compare: viewCompare, archive: viewArchive };
  app.replaceChildren(...[(views[v] || viewDay)()].flat());
  updateCmpUI();
}
function renderKeepFocus(input) {
  const pos = input.selectionStart;
  const label = input.getAttribute("aria-label");
  render();
  const el = app.querySelector(`input[aria-label="${label}"]`);
  if (el) { el.focus(); el.setSelectionRange(pos, pos); }
}
async function ensurePapers() {
  if (S.papers) return;
  S.papers = await getJSON("data/papers.json").catch(() => ({ papers: {}, topics: [], default_topics: [] }));
  let saved = null;
  try { saved = JSON.parse(store("radar-topics") || "null"); } catch { saved = null; }
  S.topics = new Set(Array.isArray(saved) ? saved : (S.papers.default_topics || []));
}
let lastView = null;
async function route() {
  const p = new URLSearchParams(location.hash.slice(1));
  const v = p.get("v") || "day";
  if (v === "day") {
    const date = p.get("d") || (S.index.days[0] && S.index.days[0].date);
    if (date && (!S.day || S.day.date !== date)) {
      S.day = await getJSON(`data/days/${date}.json`).catch(() => null);
      S.show = {};
      window.scrollTo(0, 0);
    }
  }
  if (v === "day" || v === "papers") await ensurePapers();
  if (v === "papers" && S.pOpen.size) await ensurePaperDetails();
  if (v === "day" && S.day) trackFresh(S.day);
  if (v === "embeddings" && !S.emb) S.emb = await getJSON("data/embeddings.json").catch(() => null);
  if (v !== lastView) { window.scrollTo(0, 0); lastView = v; }
  render();
  if (p.get("m")) openModel(p.get("m"));
  else if (modal.open) modal.close();
}

function renderFoot() {
  const st = S.index.status || {};
  const src = Object.entries(st.sources || {});
  const bad = src.filter(([, x]) => !x.ok);
  const llm = st.llm || {};
  const upd = S.index.updated ? new Date(S.index.updated).toLocaleString("it-IT", { dateStyle: "medium", timeStyle: "short" }) : "?";
  document.getElementById("foot").replaceChildren(
    h("div", {}, `Aggiornato ${upd} · ${src.length - bad.length}/${src.length} fonti ok`,
      llm.calls ? ` · LLM: ${llm.calls} chiamate${llm.failures ? ` (${llm.failures} fallite)` : ""}${llm.cost_usd ? `, $${llm.cost_usd.toFixed(3)}` : ""}` : " · LLM non attivo",
      st.duration_s ? ` · ${st.duration_s}s` : ""),
    h("details", {}, h("summary", {}, "Dettagli esecuzione"),
      h("div", {}, src.map(([k, x]) => h("span", { class: "src" + (x.ok ? "" : " bad"), title: x.error || "" }, `${x.ok ? "✓" : "✗"} ${k}${x.n != null ? ` (${x.n})` : ""}`))),
      Object.keys(llm.models || {}).length ? h("div", {}, "Modelli LLM usati: ", Object.entries(llm.models).map(([m, n]) => `${m} ×${n}`).join(", ")) : null,
      (llm.log || []).length ? h("div", { class: "small" }, "Log LLM: ", llm.log.join(" | ")) : null));
}

function setupTheme() {
  const btn = document.getElementById("theme");
  const apply = (t) => { if (t === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", t); btn.title = `Tema: ${t}`; };
  let t = store("radar-theme") || "auto";
  apply(t);
  btn.addEventListener("click", () => { t = t === "auto" ? "dark" : t === "dark" ? "light" : "auto"; store("radar-theme", t); apply(t); });
}

async function init() {
  setupTheme();
  S.seenStart = loadSeen(); // fotografia di ciò che avevi già visto, prima di questa visita
  await finishOpenRouter();
  try {
    S.index = await getJSON("data/index.json");
  } catch {
    app.replaceChildren(h("div", { class: "empty" }, h("h2", {}, "Nessun dato ancora"), h("p", {}, "Esegui ", h("code", {}, "python3 -m radar run"), " oppure il workflow GitHub.")));
    return;
  }
  S.index.site = S.index.site || { advice_model: "anthropic/claude-haiku-5.5", profile: "" };
  const hw = S.index.hardware;
  document.getElementById("gpu").textContent = `${hw.gpu_name} · ${hw.vram_gb} GB`;
  const md = await getJSON("data/models.json").catch(() => ({ models: {} }));
  S.models = md.models || {};
  for (const [a, t] of Object.entries(md.aliases || {})) {
    if (S.models[t] && !S.models[a]) Object.defineProperty(S.models, a, { value: S.models[t], enumerable: false });
  }
  renderFoot();
  window.addEventListener("hashchange", route);
  route();
}
init();
