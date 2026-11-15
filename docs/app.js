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
  emb: null, eBench: 0, eSort: "rank", eDir: 1, eQ: "", eOpen: false, eGpu: false, eLang: false, eSize: "all",
};

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
function kindBadge(e) { return h("span", { class: `badge ${e.kind}` }, KIND[e.kind] || e.kind); }
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
  if (e.aa && e.aa.intelligence != null) c.push(h("span", { class: "chip aa", title: "Artificial Analysis Intelligence Index" }, "AA ", h("b", {}, fmtNum(e.aa.intelligence))));
  return h("div", { class: "chips" }, c);
}

function verdict(e) {
  const v = e.vram;
  if (v) return h("div", { class: `verdict ${v.level}` }, h("i", { class: "dotv" }), h("span", {}, v.text));
  if (e.kind === "closed") {
    const p = e.pricing && e.pricing.in != null ? `$${fmtPrice(e.pricing.in)} in / $${fmtPrice(e.pricing.out)} out per M token` : "prezzo n/d";
    return h("div", { class: "verdict api" }, h("i", { class: "dotv" }), h("span", {}, `Solo API · ${p}`));
  }
  if (e.kind === "announced") return h("div", { class: "verdict none" }, h("i", { class: "dotv" }), h("span", {}, "Annunciato: pesi/API non ancora rilevati"));
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

function modelCard(e, ref) {
  const desc = (e.deep && e.deep.tldr) || e.description;
  const open = () => { location.hash = openModelHash(e.id); };
  const fresh = S.fresh.has(e.id);
  return h("article", { class: "mcard" + (fresh ? " fresh" : ""), tabindex: "0", role: "button", "aria-label": `Dettagli ${e.name}`, onclick: open,
      onkeydown: (ev) => { if (ev.key === "Enter" && ev.target === ev.currentTarget) open(); } },
    fresh ? h("span", { class: "fresh-tag" }, "novità per te") : null,
    h("div", { class: "mtop" }, kindBadge(e), newBadge(e, ref),
      h("span", { class: "muted" }, [e.org, fmtShort(e.released || e.first_seen)].filter(Boolean).join(" · ")),
      h("span", { class: "spacer" }), cmpBtn(e)),
    h("h3", {}, e.name),
    modelChips(e),
    verdict(e),
    newerNote(e),
    desc ? h("p", { class: "tldr" }, desc) : null,
    e.note ? h("p", { class: "note" }, e.note) : null);
}

function newsItem(it) {
  const relCls = it.rel >= 8 ? "r-high" : it.rel >= 6 ? "r-mid" : "";
  const links = [];
  if (it.comments_url && it.comments_url !== it.url) links.push(ext(it.comments_url, { class: "mlink" }, "discussione"));
  for (const [k, u] of Object.entries(it.links || {})) links.push(ext(u, { class: "mlink" }, k === "arxiv" ? "arXiv" : k === "github" ? "GitHub" : "progetto"));
  for (const id of it.models || []) {
    const m = S.models[id];
    if (m) links.push(h("a", { class: "mlink", href: openModelHash(id) }, "⬡ ", m.name));
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
function paperCard(p) {
  const imp = p.imp != null ? p.imp : "–";
  const relCls = p.imp >= 8 ? "r-high" : p.imp >= 6 ? "r-mid" : "";
  const L = p.links || {};
  const links = [L.arxiv && ext(L.arxiv, { class: "mlink" }, "arXiv"), L.pdf && ext(L.pdf, { class: "mlink" }, "PDF"),
    L.github && ext(L.github, { class: "mlink" }, "GitHub"), L.project && ext(L.project, { class: "mlink" }, "progetto")].filter(Boolean);
  const meta = [h("span", {}, p.source), p.published ? h("span", {}, fmtShort(p.published)) : null,
    p.upvotes ? h("span", { title: "voti su Hugging Face Papers" }, `▲ ${p.upvotes}`) : null,
    p.authors && p.authors.length ? h("span", { class: "authors" }, p.authors.slice(0, 3).join(", ") + (p.authors.length > 3 ? " et al." : "")) : null,
    p.org ? h("span", {}, p.org) : null];
  const fresh = S.fresh.has("p:" + p.id);
  return h("article", { class: "item" + (fresh ? " fresh" : "") },
    fresh ? h("span", { class: "fresh-tag" }, "novità per te") : null,
    h("div", { class: `rel ${relCls}`, title: "Importanza stimata (0-10)" }, imp),
    h("div", { class: "body" },
      h("div", { class: "meta" }, meta),
      h("h3", {}, ext(p.url, p.title_it || p.title)),
      p.title_it && p.title_it !== p.title ? h("div", { class: "orig" }, p.title) : null,
      p.summary_it ? h("p", {}, p.summary_it) : null,
      p.why ? h("p", { class: "why" }, "→ ", p.why) : null,
      h("div", { class: "links" }, (p.topics || []).map((t) => h("button", { class: "tchip" + (S.topics && S.topics.has(t) ? " on" : ""), type: "button",
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
function grid(key, list, ref, limit = 9) {
  const shown = S.show[key] ? list : list.slice(0, limit);
  return [h("div", { class: "mgrid" }, shown.map((e) => modelCard(e, ref))),
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
  if (newModels.length) {
    out.push(h("section", { class: "section" },
      h("div", { class: "section-head" }, h("h2", {}, "Nuovi modelli"), h("span", { class: "muted" }, `${newModels.length} · usciti negli ultimi ${S.index.new_model_days || 14} giorni`)),
      grid("new", newModels, d.date)));
  }
  if (cited.length) {
    out.push(h("section", { class: "section" },
      h("div", { class: "section-head" }, h("h2", {}, "Citati oggi"), h("span", { class: "muted" }, "modelli già usciti di cui si parla nelle notizie")),
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
      out.push(h("section", { class: "section" },
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
  if (!S.show.items) items = items.slice(0, 40);
  const chip = (key, label, n) => h("button", { class: "fchip" + (S.cat === key ? " on" : ""), onclick: () => { S.cat = key; S.show.items = false; render(); } },
    label, n != null ? h("span", { class: "n" }, n) : null);
  out.push(h("section", { class: "section" },
    h("div", { class: "section-head" }, h("h2", {}, "Notizie"), h("span", { class: "muted" }, `${d.items.length} selezionate su ${d.stats ? d.stats.new : "?"} nuove`)),
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
  let rows = Object.values(S.models).filter((e) => e.aa && (e.aa.intelligence != null || e.aa.coding != null || e.aa.agentic != null));
  const total = rows.length;
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
  const bar = (v, max, cls) => v == null ? h("span", { class: "muted" }, "—") :
    h("div", { class: "ibar " + (cls || "") }, h("i", { style: `width:${Math.max(2, v / max * 100)}%` }), h("b", {}, fmtNum(v)));
  return [
    h("div", { class: "section-head" }, h("h2", {}, "Classifica"),
      h("span", { class: "muted" }, `${total} modelli con indici di Artificial Analysis · clicca le colonne per ordinare`)),
    h("div", { class: "filters" },
      chip("all", "Tutti"), chip("open", "Pesi aperti"), chip("gpu", `Girano sulla ${gpu}`), chip("api", "Solo API"), chip("new", "Nuovi"),
      h("input", { class: "search", type: "search", placeholder: "Cerca…", value: S.lbQ, "aria-label": "Cerca in classifica",
        oninput: (ev) => { S.lbQ = ev.target.value; renderKeepFocus(ev.target); } }),
      latestToggle(hidden)),
    h("div", { class: "tbl-wrap card" }, h("table", { class: "tbl lb" },
      h("thead", {}, h("tr", {}, h("th", { class: "num" }, "#"), th("name", "Modello"), th("intelligence", "Intelligence", "wide"),
        th("coding", "Coding"), th("agentic", "Agentic"), th("price", "$ in / out"), th("context", "Contesto"),
        h("th", {}, gpu), th("released", "Uscita"), h("th", {}, ""))),
      h("tbody", {}, rows.map((e, i) => h("tr", {},
        h("td", { class: "num muted" }, i + 1),
        h("td", {}, h("a", { href: openModelHash(e.id), class: "lb-name" }, e.name), h("div", { class: "small" }, kindBadge(e), " ", e.org || "", newBadge(e) ? " · nuovo" : "")),
        h("td", { class: "wide" }, bar(e.aa.intelligence, maxI)),
        h("td", { class: "num" }, fmtNum(e.aa.coding)), h("td", { class: "num" }, fmtNum(e.aa.agentic)),
        h("td", { class: "num" }, e.pricing && e.pricing.in != null ? `${fmtPrice(e.pricing.in)} / ${fmtPrice(e.pricing.out)}` : "—"),
        h("td", { class: "num" }, fmtCtx(e.context) || "—"),
        h("td", {}, e.vram ? [h("span", { class: `lvl ${e.vram.level}`, title: vramTip(e.vram) }, e.vram.label),
          e.vram.quantized ? h("div", { class: "small" }, "quantizzato") : null] : h("span", { class: "muted" }, e.kind === "closed" ? "API" : "—")),
        h("td", { class: "num" }, fmtShort(e.released) || "—"),
        h("td", {}, cmpBtn(e))))))),
    h("p", { class: "small source-note" },
      `Colonna ${gpu}: quantizzazione consigliata (la migliore che lascia almeno ${S.index.hardware.margin_gb ?? 1.5} GB liberi) e contesto massimo; `,
      "«BF16 nativo» = gira senza quantizzare. Passa il mouse per le alternative, clicca il modello per il calcolatore completo. ",
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
  const inWin = Object.values(P.papers).filter((p) => (p.published || p.first_seen || "").slice(0, 10) >= cutoff);
  const counts = {};
  for (const p of inWin) for (const t of p.topics || []) counts[t] = (counts[t] || 0) + 1;
  const q = S.pQ.trim().toLowerCase();
  let list = inWin.filter((p) => (!S.topics.size || (p.topics || []).some((t) => S.topics.has(t))) &&
    (!q || `${p.title} ${p.title_it || ""} ${p.summary_it || ""} ${(p.authors || []).join(" ")}`.toLowerCase().includes(q)));
  const sorters = { score: (a, b) => b.score - a.score, votes: (a, b) => (b.upvotes || 0) - (a.upvotes || 0) || b.score - a.score,
    recent: (a, b) => (b.published || "").localeCompare(a.published || "") };
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
        [["score", "Più rilevanti"], ["votes", "Più votati"], ["recent", "Più recenti"]].map(([v, l]) => h("option", { value: v, selected: S.pSort === v }, l))),
      h("input", { class: "search", type: "search", placeholder: "Cerca titolo, autore…", value: S.pQ, "aria-label": "Cerca nei paper",
        oninput: (ev) => { S.pQ = ev.target.value; renderKeepFocus(ev.target); } }),
      h("span", { class: "muted small" }, `${total} paper`)),
    list.length ? h("div", { class: "items" }, list.map(paperCard)) : h("p", { class: "empty" }, "Nessun paper con questi filtri."),
    total > list.length ? h("div", { class: "more" }, h("button", { class: "btn", onclick: () => { S.pLimit += 40; render(); } }, `Altri (${total - list.length})`)) : null,
    h("p", { class: "small source-note" }, "Fonti: Hugging Face Daily Papers, arXiv (ricerca per argomento), Alignment Forum. Argomenti, importanza e riassunti sono assegnati da un LLM: verifica sempre sul paper."),
  ];
}

function viewArchive() {
  return [
    h("div", { class: "section-head" }, h("h2", {}, "Archivio"), h("span", { class: "muted" }, `${S.index.days.length} giorni`)),
    h("div", { class: "days" }, S.index.days.map((d) => h("a", { class: "dayrow", href: `#v=day&d=${d.date}` },
      h("span", { class: "d" }, fmtDate(d.date)), h("span", { class: "hl" }, d.headline || h("span", { class: "muted" }, "—")),
      h("span", { class: "c" }, `${d.n_items} notizie · ${d.n_models} modelli`)))),
  ];
}

/* ───────────── embedding (leaderboard MTEB) ───────────── */
const EMB_TYPES = ["Retrieval", "Reranking", "STS", "Classification", "Clustering", "PairClassification", "InstructionReranking", "BitextMining", "MultilabelClassification"];
const TYPE_IT = { Retrieval: "Retrieval", Reranking: "Reranking", STS: "STS", Classification: "Classific.", Clustering: "Clustering",
  PairClassification: "Coppie", InstructionReranking: "Rerank istr.", BitextMining: "Bitext", MultilabelClassification: "Multilabel" };
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
  const types = EMB_TYPES.filter((t) => (B.types || []).includes(t)).slice(0, 6);
  const langLabel = LANG_IT[E.language] || E.language;
  const q = S.eQ.trim().toLowerCase();
  const sizeOk = (p) => S.eSize === "all" || (p != null && (S.eSize === "s" ? p < 1 : S.eSize === "m" ? p >= 1 && p < 5 : S.eSize === "l" ? p >= 5 && p < 10 : p >= 10));
  let rows = B.rows.map((r) => ({ ...r, meta: E.models[r.m] || { name: r.m } }));
  const hiddenOld = rows.filter((r) => r.meta.newer).length;
  rows = rows.filter((r) => {
    const m = r.meta;
    if (S.latestOnly && m.newer) return false;
    if (S.eOpen && !m.open) return false;
    if (S.eGpu) { const f = embFits(m); if (!m.open || !f || !f.fits) return false; }
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
      onclick: () => { S.eBench = i; if (S.eSort.startsWith("t:") || (S.eSort === "lang" && !E.benchmarks[i].language)) { S.eSort = "rank"; S.eDir = 1; } render(); } }, b.label))),
    h("div", { class: "filters" },
      toggle("eOpen", "Pesi aperti"), toggle("eGpu", `Girano sulla ${gpu}`), toggle("eLang", `Supportano l'${langLabel.toLowerCase()}`),
      h("select", { class: "search", "aria-label": "Dimensione", onchange: (ev) => { S.eSize = ev.target.value; render(); } },
        [["all", "Ogni dimensione"], ["s", "< 1B"], ["m", "1–5B"], ["l", "5–10B"], ["xl", "> 10B"]].map(([v, l]) => h("option", { value: v, selected: S.eSize === v }, l))),
      h("input", { class: "search", type: "search", placeholder: "Cerca…", value: S.eQ, "aria-label": "Cerca embedding",
        oninput: (ev) => { S.eQ = ev.target.value; renderKeepFocus(ev.target); } }),
      latestToggle(hiddenOld)),
    h("div", { class: "tbl-wrap card" }, h("table", { class: "tbl lb" },
      h("thead", {}, h("tr", {}, th("rank", "#", "Posizione nella leaderboard ufficiale"), th("name", "Modello"), th("mean", "Media", "Media dei task"),
        th("meanType", "Media tipi", "Media per tipo di task"), B.language ? th("lang", langLabel, `Media dei task in ${langLabel.toLowerCase()}`) : null,
        types.map((t) => th("t:" + t, TYPE_IT[t] || t)), th("params", "Param."), th("dim", "Dim."), th("maxTokens", "Max token"), th("released", "Uscita"), h("th", {}, gpu))),
      h("tbody", {}, rows.map((r) => {
        const m = r.meta;
        const fit = embFits(m);
        return h("tr", {},
          h("td", { class: "num muted" }, r.rank),
          h("td", {}, ext(m.url || `https://huggingface.co/${m.name}`, { class: "lb-name" }, m.name.split("/").pop()),
            h("div", { class: "small" }, h("span", { class: `badge ${m.open ? "open" : "closed"}` }, m.open ? "aperto" : "API"), " ", m.name.includes("/") ? m.name.split("/")[0] : "",
              embNew(m) ? " · nuovo" : "", r.zs != null && r.zs >= 0 && r.zs < 90 ? h("span", { class: "zs", title: "Percentuale di task su cui il modello NON è stato addestrato (zero-shot): sotto 100% i punteggi possono essere gonfiati" }, ` · zero-shot ${r.zs}%`) : "",
              m.newer ? h("span", { class: "newer" }, ` · superato da ${m.newer.split("/").pop()}`) : "")),
          h("td", { class: "wide" }, h("div", { class: "ibar" }, h("i", { style: `width:${Math.max(2, (r.mean || 0) / maxMean * 100)}%` }), h("b", {}, fmtNum(r.mean)))),
          h("td", { class: "num" }, fmtNum(r.meanType)),
          B.language ? h("td", { class: "num" }, fmtNum(r.lang)) : null,
          types.map((t) => h("td", { class: "num" }, fmtNum(r.types[t]))),
          h("td", { class: "num" }, m.params ? fmtParams(m.params) : "—"), h("td", { class: "num" }, m.dim || "—"),
          h("td", { class: "num" }, m.maxTokens ? fmtCtx(m.maxTokens) : "—"), h("td", { class: "num" }, fmtShort(m.released) || "—"),
          h("td", {}, m.open && fit ? h("span", { class: `lvl ${fit.fits ? "ok" : "no"}` }, `${fit.gb.toFixed(1)} GB`) : h("span", { class: "muted" }, m.open ? "—" : "API")));
      })))),
    h("p", { class: "small source-note" }, `Fonte: leaderboard ufficiale MTEB (${B.name}, ${B.total} modelli valutati; qui i primi 150 più tutti quelli recenti). `,
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
    ["Prezzo input $/M", (e) => e.pricing && e.pricing.in != null ? `$${fmtPrice(e.pricing.in)}` : "—", (e) => e.pricing && e.pricing.in, "min"],
    ["Prezzo output $/M", (e) => e.pricing && e.pricing.out != null ? `$${fmtPrice(e.pricing.out)}` : "—", (e) => e.pricing && e.pricing.out, "min"],
    ["AA Intelligence", (e) => fmtNum((e.aa || {}).intelligence), (e) => (e.aa || {}).intelligence, "max"],
    ["AA Coding", (e) => fmtNum((e.aa || {}).coding), (e) => (e.aa || {}).coding, "max"],
    ["AA Agentic", (e) => fmtNum((e.aa || {}).agentic), (e) => (e.aa || {}).agentic, "max"],
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
  const margin = hw.margin_gb ?? 1.5;
  const update = () => {
    const ctx = steps[ci];
    out.textContent = fmtCtx(ctx);
    const recDone = {};
    const groups = [["gguf", QUANTS], ["native", nativeDefs()]];
    tbody.replaceChildren(...groups.flatMap(([g, defs]) => (defs.length && g === "native" ? [h("tr", { class: "sep" },
      h("th", { colspan: 6 }, `Formati nativi della ${hw.gpu_name} (accelerazione hardware: vLLM, SGLang, TensorRT-LLM; NVFP4 anche in GGUF)`))] : [])
      .concat(defs.map(([q, bpw]) => {
      const r = vramRow(e, bpw, known ? ctx : 0, KV_TYPES[kvi][1]);
      const comfy = r.total <= hw.vram_gb - hw.reserved_gb - margin;
      const rec = comfy && !recDone[g];
      if (rec) recDone[g] = true;
      const pct = Math.min(100, r.total / hw.vram_gb * 100);
      return h("tr", { class: rec ? "rec" : "" },
        h("td", {}, h("b", {}, q)), h("td", { class: "num" }, `${r.w.toFixed(1)} GB`), h("td", { class: "num" }, `${r.total.toFixed(1)} GB`),
        h("td", {}, h("div", { class: "bar" }, h("i", { class: comfy ? "ok" : r.fits ? "tight" : "no", style: `width:${pct}%` }))),
        h("td", { class: comfy ? "fit-ok" : r.fits ? "fit-tight" : "fit-no" }, rec ? (q === "BF16" ? "★ nativo" : "★ consigliata") : comfy ? "entra" : r.fits ? "al limite" : "no"),
        h("td", { class: "num" }, r.fits && isTextModel(e) ? `${fmtTok(r.tok)} tok/s` : "—"));
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
      h("thead", {}, h("tr", {}, ["Quant", "Pesi", "Totale", `su ${hw.vram_gb} GB`, "", isTextModel(e) ? "Velocità" : ""].map((t) => h("th", {}, t)))), tbody)),
    h("p", { class: "small" }, `Stima per ${hw.gpu_name} (${hw.vram_gb} GB, ${hw.reserved_gb} GB riservati, ${hw.bandwidth_gbs} GB/s). `,
      `«Consigliata» = la quantizzazione di qualità più alta che lascia almeno ${margin} GB liberi; «al limite» = entra ma senza margine. `,
      nativeDefs().length ? "I formati nativi girano con accelerazione hardware; la velocità indicata è la stessa stima legata alla banda di memoria. " : "",
      "Bit per peso dei GGUF di llama.cpp; overhead 0,75 GB + 2% dei pesi; tok/s = generazione a contesto breve, indicativi (±25%). ",
      e.active_b ? "MoE: in VRAM vanno tutti i parametri, la velocità dipende da quelli attivi. " : "",
      a && a.linear_layers ? `Attenzione ibrida: solo ${a.full_layers + (a.sliding_layers || 0)} layer su ${a.layers} hanno KV cache che cresce col contesto. ` : "",
      a && a.mla_dim ? "MLA: KV cache compressa. " : ""));
}

function renderModel(e) {
  const d = e.deep || {};
  const secs = [];
  secs.push(h("div", { class: "md-head" },
    h("div", { class: "mtop" }, kindBadge(e), newBadge(e), h("span", { class: "muted" }, [e.org, e.released ? `uscito ${fmtShort(e.released)}` : null, e.first_seen ? `visto ${fmtShort(e.first_seen)}` : null].filter(Boolean).join(" · ")), h("span", { class: "spacer" }), cmpBtn(e)),
    h("h2", {}, e.name), modelChips(e)));
  const v = verdict(e);
  if (v) secs.push(h("div", { class: "md-sec" }, v));
  const nn = newerNote(e);
  if (nn) secs.push(nn);
  if (e.note) secs.push(h("p", { class: "note md-sec" }, e.note));
  if (d.tldr || e.description) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "In breve"), h("p", {}, d.tldr || e.description)));
  if (d.features && d.features.length) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Caratteristiche"), h("ul", {}, d.features.map((f) => h("li", {}, f)))));
  const facts = [["Parametri dichiarati", d.params], ["Contesto dichiarato", d.context], ["Disponibilità", d.availability],
    ["Knowledge cutoff", e.knowledge_cutoff], ["Gated", e.gated ? "sì (accetta la licenza su HF)" : null]].filter(([, x]) => x);
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
  if (d.local_tips) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Consigli per l'uso locale"), h("p", {}, d.local_tips)));
  if (e.mentions && e.mentions.length) {
    secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Ne parlano"),
      h("ul", {}, e.mentions.map((m) => h("li", {}, ext(m.url, m.title), h("span", { class: "small" }, ` — ${m.source}, ${fmtShort(m.date)}`))))));
  }
  const L = e.links || {};
  const links = [L.hf && ["Hugging Face", L.hf], L.openrouter && ["OpenRouter", L.openrouter], L.announcement && ["Annuncio", L.announcement]].filter(Boolean);
  if (links.length) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Link"), h("div", { class: "chips" }, links.map(([n, u]) => ext(u, { class: "mlink" }, n)))));
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
    h("div", { class: "md-sec" }, h("h3", {}, "OpenRouter (per il consiglio nel confronto)"),
      h("p", {}, key ? `Collegato · chiave …${key.slice(-4)}` : "Non collegato."),
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
      h("p", {}, `${S.index.hardware.gpu_name}, ${S.index.hardware.vram_gb} GB VRAM, ${S.index.hardware.bandwidth_gbs} GB/s, ${S.index.hardware.system_ram_gb} GB RAM. Si cambia in config.toml.`)));
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
  S.models = (await getJSON("data/models.json").catch(() => ({ models: {} }))).models || {};
  renderFoot();
  window.addEventListener("hashchange", route);
  route();
}
init();
