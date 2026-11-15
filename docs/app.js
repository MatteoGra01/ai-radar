"use strict";

/* ───────────── utilità ───────────── */
const GiB = 2 ** 30;
const QUANTS = [["BF16", 16], ["Q8_0", 8.5], ["Q6_K", 6.56], ["Q5_K_M", 5.69], ["Q4_K_M", 4.85], ["IQ4_XS", 4.25], ["Q3_K_M", 3.91], ["IQ2_M", 2.7]];
const KV_TYPES = [["F16", 2], ["Q8_0", 1.0625], ["Q4_0", 0.5625]];
const CATS = { modello: "Modelli", feature: "Feature", tool: "Tool", paper: "Paper", guida: "Guide e analisi", industria: "Industria", altro: "Altro" };
const CAT1 = { modello: "Modello", feature: "Feature", tool: "Tool", paper: "Paper", guida: "Guida", industria: "Industria", altro: "Altro" };
const KIND = { open: "Pesi aperti", closed: "API", announced: "Annunciato", placeholder: "Senza pesi" };
const S = { index: null, models: {}, day: null, cat: "tutte", minRel: 0, q: "", showAllModels: false, showAllItems: false,
  mFilter: "all", mSort: "recent", mQ: "", mLimit: 60 };

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
const TEXT_PIPES = new Set(["text-generation", "image-text-to-text", "any-to-any", "text2text-generation", "conversational"]);
const isTextModel = (e) => !e.pipeline || TEXT_PIPES.has(e.pipeline);
const fmtTok = (t) => t >= 500 ? "500+" : `~${Math.round(t)}`;
function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch { return null; } }

const fmtDate = (d) => new Date(d + "T12:00:00").toLocaleDateString("it-IT", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
const fmtShort = (d) => d ? new Date(d.slice(0, 10) + "T12:00:00").toLocaleDateString("it-IT", { day: "numeric", month: "short" }) : "";
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
function vramRow(e, bpw, ctx, kvb) {
  const hw = S.index.hardware;
  const w = e.params_b * 1e9 * bpw / 8 / GiB;
  const total = w + kvBytes(e.arch, ctx, kvb) / GiB + 0.75 + 0.02 * w;
  const act = e.active_b || e.params_b;
  const moe = e.active_b && e.active_b < e.params_b * 0.7;
  const eff = moe ? Math.min(0.7, 0.15 + 0.03 * act) : 0.7;
  const tok = eff * hw.bandwidth_gbs * 1e9 / (act * 1e9 * bpw / 8 + kvBytes(e.arch, 4096, kvb));
  return { w, total, fits: total <= hw.vram_gb - hw.reserved_gb, tok };
}

/* ───────────── componenti ───────────── */
function kindBadge(e) { return h("span", { class: `badge ${e.kind}` }, KIND[e.kind] || e.kind); }

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
  if (e.aa && e.aa.evals) {
    const ii = Object.entries(e.aa.evals).find(([k]) => /intelligence/i.test(k));
    if (ii) c.push(h("span", { class: "chip", title: "Artificial Analysis Intelligence Index" }, `AA ${+ii[1].toFixed(1)}`));
  }
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

function openModelHash(id) {
  const p = new URLSearchParams(location.hash.slice(1));
  p.set("m", id);
  return "#" + p.toString();
}

function modelCard(e) {
  const desc = (e.deep && e.deep.tldr) || e.description;
  return h("article", { class: "mcard", tabindex: "0", role: "button", "aria-label": `Dettagli ${e.name}`,
      onclick: () => { location.hash = openModelHash(e.id); },
      onkeydown: (ev) => { if (ev.key === "Enter") location.hash = openModelHash(e.id); } },
    h("div", { class: "mtop" }, kindBadge(e), h("span", { class: "muted" }, [e.org, fmtShort(e.released || e.first_seen)].filter(Boolean).join(" · "))),
    h("h3", {}, e.name),
    modelChips(e),
    verdict(e),
    desc ? h("p", { class: "tldr" }, desc) : null,
    e.note ? h("p", { class: "note" }, e.note) : null);
}

function newsItem(it) {
  const relCls = it.rel >= 8 ? "r-high" : it.rel >= 6 ? "r-mid" : "";
  const links = [];
  if (it.comments_url && it.comments_url !== it.url) links.push(ext(it.comments_url, { class: "mlink" }, "discussione"));
  for (const [k, u] of Object.entries(it.links || {})) links.push(h("a", { class: "mlink", href: u, target: "_blank", rel: "noopener noreferrer" }, k === "arxiv" ? "arXiv" : k === "github" ? "GitHub" : "progetto"));
  for (const id of it.models || []) {
    const m = S.models[id];
    if (m) links.push(h("a", { class: "mlink", href: openModelHash(id) }, "⬡ ", m.name));
  }
  const meta = [h("span", { class: `cat cat-${it.cat}` }, CAT1[it.cat] || it.cat),
    h("span", {}, it.source + (it.also_on && it.also_on.length ? ` +${it.also_on.length}` : ""))];
  if (it.engagement) meta.push(h("span", {}, it.source === "GitHub" ? `★ ${it.engagement}` : `${it.engagement} punti`));
  if (it.published) meta.push(h("span", {}, ago(it.published)));
  return h("article", { class: "item" },
    h("div", { class: `rel ${relCls}`, title: it.llm ? "Rilevanza per te (0-10)" : "Punteggio euristico (senza LLM)" }, it.rel),
    h("div", { class: "body" },
      h("div", { class: "meta" }, meta),
      h("h3", {}, ext(it.url, it.title)),
      it.title !== it.title_orig ? h("div", { class: "orig" }, it.title_orig) : null,
      it.summary ? h("p", {}, it.summary) : null,
      it.why ? h("p", { class: "why" }, "→ ", it.why) : null,
      links.length ? h("div", { class: "links" }, links) : null));
}

/* ───────────── viste ───────────── */
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
  } else if (!(S.index.status && S.index.status.llm && S.index.status.llm.calls)) {
    out.push(h("div", { class: "hint" }, "Riassunti, traduzioni e benchmark richiedono un LLM: imposta il secret ",
      h("code", {}, "OPENROUTER_API_KEY"), " (gratuito) e rilancia il workflow. Per ora vedi notizie e modelli con punteggio euristico."));
  }

  // modelli
  const models = (d.models || []).map((id) => S.models[id]).filter(Boolean);
  if (models.length) {
    const shown = S.showAllModels ? models : models.slice(0, 9);
    out.push(h("section", { class: "section" },
      h("div", { class: "section-head" }, h("h2", {}, "Modelli nuovi e citati"), h("span", { class: "muted" }, `${models.length} · clicca per specifiche, benchmark e calcolo VRAM`)),
      h("div", { class: "mgrid" }, shown.map(modelCard)),
      models.length > 9 ? h("div", { class: "more" }, h("button", { class: "btn", onclick: () => { S.showAllModels = !S.showAllModels; render(); } },
        S.showAllModels ? "Mostra meno" : `Mostra tutti (${models.length})`)) : null));
  }

  // notizie
  const counts = {};
  for (const it of d.items) counts[it.cat] = (counts[it.cat] || 0) + 1;
  const q = S.q.trim().toLowerCase();
  let items = d.items.filter((it) => (S.cat === "tutte" || it.cat === S.cat) && it.rel >= S.minRel &&
    (!q || `${it.title} ${it.title_orig} ${it.summary} ${it.source}`.toLowerCase().includes(q)));
  const total = items.length;
  if (!S.showAllItems) items = items.slice(0, 40);
  const chip = (key, label, n) => h("button", { class: "fchip" + (S.cat === key ? " on" : ""), onclick: () => { S.cat = key; S.showAllItems = false; render(); } },
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
    total > items.length ? h("div", { class: "more" }, h("button", { class: "btn", onclick: () => { S.showAllItems = true; render(); } }, `Mostra tutte (${total})`)) : null));
  return out;
}

function viewModels() {
  const all = Object.values(S.models);
  const q = S.mQ.trim().toLowerCase();
  let list = all.filter((e) => {
    if (S.mFilter === "open" && e.kind !== "open") return false;
    if (S.mFilter === "api" && e.kind !== "closed" && !e.or_id) return false;
    if (S.mFilter === "gpu" && !(e.vram && e.vram.level === "ok")) return false;
    if (S.mFilter === "announced" && !["announced", "placeholder"].includes(e.kind)) return false;
    return !q || `${e.name} ${e.org} ${e.id} ${e.pipeline || ""}`.toLowerCase().includes(q);
  });
  const sorters = {
    recent: (a, b) => (b.released || b.first_seen || "").localeCompare(a.released || a.first_seen || ""),
    score: (a, b) => (b.score || 0) - (a.score || 0),
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
      chip("all", "Tutti"), chip("gpu", `Girano sulla ${S.index.hardware.gpu_name}`), chip("open", "Pesi aperti"), chip("api", "API"), chip("announced", "Annunciati / senza pesi"),
      h("select", { class: "search", "aria-label": "Ordina", onchange: (ev) => { S.mSort = ev.target.value; render(); } },
        [["recent", "Più recenti"], ["score", "Più rilevanti"], ["likes", "Più popolari"], ["params", "Più grandi"]].map(([v, l]) => h("option", { value: v, selected: S.mSort === v }, l))),
      h("input", { class: "search", type: "search", placeholder: "Cerca modello…", value: S.mQ, "aria-label": "Cerca modello",
        oninput: (ev) => { S.mQ = ev.target.value; renderKeepFocus(ev.target); } })),
    list.length ? h("div", { class: "mgrid" }, list.map(modelCard)) : h("p", { class: "empty" }, "Nessun modello."),
    total > list.length ? h("div", { class: "more" }, h("button", { class: "btn", onclick: () => { S.mLimit += 60; render(); } }, `Altri (${total - list.length})`)) : null,
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

/* ───────────── dettaglio modello ───────────── */
function vramCalc(e) {
  const hw = S.index.hardware;
  const a = e.arch;
  const kvKnown = !!(a && (a.mla_dim || (a.kv_heads && a.head_dim)));
  const maxPos = (a && a.max_ctx) || e.context || 131072;
  const steps = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576].filter((c) => c <= maxPos);
  if (!steps.length || steps[steps.length - 1] < maxPos) steps.push(maxPos);
  let ci = Math.max(0, steps.findIndex((c) => c >= Math.min(hw.ref_context, maxPos)));
  let kvi = 0;
  const out = h("output", {});
  const tbody = h("tbody", {});
  const update = () => {
    const ctx = steps[ci];
    out.textContent = fmtCtx(ctx);
    tbody.replaceChildren(...QUANTS.map(([q, bpw]) => {
      const r = vramRow(e, bpw, kvKnown ? ctx : 0, KV_TYPES[kvi][1]);
      const pct = Math.min(100, r.total / hw.vram_gb * 100);
      return h("tr", {},
        h("td", {}, h("b", {}, q)), h("td", { class: "num" }, `${r.w.toFixed(1)} GB`), h("td", { class: "num" }, `${r.total.toFixed(1)} GB`),
        h("td", {}, h("div", { class: "bar" }, h("i", { class: r.fits ? "ok" : "no", style: `width:${pct}%` }))),
        h("td", { class: r.fits ? "fit-ok" : "fit-no" }, r.fits ? "entra" : "no"),
        h("td", { class: "num" }, r.fits && isTextModel(e) ? `${fmtTok(r.tok)} tok/s` : "—"));
    }));
  };
  const slider = h("input", { type: "range", min: 0, max: steps.length - 1, value: ci, "aria-label": "Contesto",
    oninput: (ev) => { ci = +ev.target.value; update(); } });
  const kvSel = h("select", { class: "search", "aria-label": "Tipo KV cache", onchange: (ev) => { kvi = +ev.target.value; update(); } },
    KV_TYPES.map(([n], i) => h("option", { value: i }, `KV cache ${n}`)));
  update();
  return h("div", {},
    h("div", { class: "calc-ctl" },
      kvKnown ? h("label", {}, "Contesto ", slider, " ", out) : h("span", { class: "small" }, "Architettura non disponibile: stima dei soli pesi."),
      kvKnown ? kvSel : null),
    h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" },
      h("thead", {}, h("tr", {}, ["Quant", "Pesi", "Totale", `su ${hw.vram_gb} GB`, "", isTextModel(e) ? "Velocità" : ""].map((t) => h("th", {}, t)))), tbody)),
    h("p", { class: "small" }, `Stima per ${hw.gpu_name} (${hw.vram_gb} GB, ${hw.reserved_gb} GB riservati, ${hw.bandwidth_gbs} GB/s). `,
      "Bit per peso dei GGUF di llama.cpp; overhead 0,75 GB + 2% dei pesi; tok/s = generazione a contesto breve, indicativi (±25%). ",
      e.active_b ? "MoE: in VRAM vanno tutti i parametri, la velocità dipende da quelli attivi. " : "",
      a && a.linear_layers ? `Attenzione ibrida: solo ${a.full_layers + (a.sliding_layers || 0)} layer su ${a.layers} hanno KV cache che cresce col contesto. ` : "",
      a && a.mla_dim ? "MLA: KV cache compressa. " : ""));
}

function renderModel(e) {
  const d = e.deep || {};
  const secs = [];
  secs.push(h("div", { class: "md-head" },
    h("div", { class: "mtop" }, kindBadge(e), h("span", { class: "muted" }, [e.org, e.released ? `rilasciato ${fmtShort(e.released)}` : null, e.first_seen ? `visto ${fmtShort(e.first_seen)}` : null].filter(Boolean).join(" · "))),
    h("h2", {}, e.name), modelChips(e)));
  const v = verdict(e);
  if (v) secs.push(h("div", { class: "md-sec" }, v));
  if (e.note) secs.push(h("p", { class: "note md-sec" }, e.note));
  if (d.tldr || e.description) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "In breve"), h("p", {}, d.tldr || e.description)));
  if (d.features && d.features.length) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Caratteristiche"), h("ul", {}, d.features.map((f) => h("li", {}, f)))));
  const facts = [["Parametri dichiarati", d.params], ["Contesto dichiarato", d.context], ["Disponibilità", d.availability],
    ["Knowledge cutoff", e.knowledge_cutoff], ["Gated", e.gated ? "sì (accetta la licenza su HF)" : null]].filter(([, x]) => x);
  if (facts.length) secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Dati"), h("div", { class: "kv" }, facts.map(([k, x]) => h("div", {}, h("span", {}, k + ": "), x)))));
  if (d.benchmarks && d.benchmarks.length) {
    secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Benchmark dichiarati"),
      h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" },
        h("thead", {}, h("tr", {}, h("th", {}, "Benchmark"), h("th", {}, e.name), h("th", {}, "Confronto"))),
        h("tbody", {}, d.benchmarks.map((b) => h("tr", {}, h("td", {}, b.name), h("td", { class: "num" }, h("b", {}, b.score)),
          h("td", { class: "vs" }, (b.vs || []).map((x) => `${x.model} ${x.score}`).join(" · "))))))),
      d.source ? h("p", { class: "small" }, "Fonte: ", ext(d.source, host(d.source)), d.by ? ` · estratto da ${d.by}` : "") : null));
  }
  if (e.aa && e.aa.evals) {
    secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Artificial Analysis"),
      h("div", { class: "kv" }, Object.entries(e.aa.evals).map(([k, x]) => h("div", {}, h("span", {}, k.replace(/_/g, " ") + ": "), typeof x === "number" ? +x.toFixed(3) : x))),
      e.aa.url ? h("p", { class: "small" }, ext(e.aa.url, "Pagina su artificialanalysis.ai")) : null));
  }
  if (e.params_b && e.kind !== "closed") secs.push(h("div", { class: "md-sec" }, h("h3", {}, `Gira sulla tua ${S.index.hardware.gpu_name}?`), vramCalc(e)));
  if (e.quants && e.quants.length) {
    secs.push(h("div", { class: "md-sec" }, h("h3", {}, "Quantizzazioni pronte"),
      h("div", { class: "chips" }, e.quants.map((q) => ext(`https://huggingface.co/${q.id}`, { class: "mlink" }, `${q.id} · ${q.format}`)))));
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

/* ───────────── routing e render ───────────── */
const app = document.getElementById("app");
function currentView() { return new URLSearchParams(location.hash.slice(1)).get("v") || "day"; }

function render() {
  const v = currentView();
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.view === v));
  const content = v === "models" ? viewModels() : v === "archive" ? viewArchive() : viewDay();
  app.replaceChildren(...[content].flat());
}
function renderKeepFocus(input) {
  const pos = input.selectionStart;
  const label = input.getAttribute("aria-label");
  render();
  const el = app.querySelector(`input[aria-label="${label}"]`);
  if (el) { el.focus(); el.setSelectionRange(pos, pos); }
}

async function route() {
  const p = new URLSearchParams(location.hash.slice(1));
  const v = p.get("v") || "day";
  if (v === "day") {
    const date = p.get("d") || (S.index.days[0] && S.index.days[0].date);
    if (date && (!S.day || S.day.date !== date)) {
      S.day = await getJSON(`data/days/${date}.json`).catch(() => null);
      S.showAllModels = false; S.showAllItems = false;
      window.scrollTo(0, 0);
    }
  }
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
  try {
    S.index = await getJSON("data/index.json");
  } catch {
    app.replaceChildren(h("div", { class: "empty" }, h("h2", {}, "Nessun dato ancora"), h("p", {}, "Esegui ", h("code", {}, "python3 -m radar run"), " oppure il workflow GitHub.")));
    return;
  }
  const hw = S.index.hardware;
  document.getElementById("gpu").textContent = `${hw.gpu_name} · ${hw.vram_gb} GB`;
  S.models = (await getJSON("data/models.json").catch(() => ({ models: {} }))).models || {};
  renderFoot();
  window.addEventListener("hashchange", route);
  route();
}
init();
