# AI Radar

Rassegna quotidiana **personale** su nuovi modelli AI, benchmark, paper e nuove funzionalità, con verdetto
"gira sulla mia RTX 5090?" calcolato dall'architettura reale del modello.
Costo: ~1 $/mese con i modelli economici di OpenRouter (0 € con i soli modelli gratuiti).

## Sezioni del sito

- **Oggi** — editoriale, nuovi modelli, modelli citati, paper sui tuoi argomenti, notizie filtrabili.
  Le **novità dall'ultima visita** sono evidenziate (il browser ricorda cosa hai già visto) e c'è il filtro
  "solo novità". In prima pagina compaiono **solo le ultime versioni**: se di un modello esiste già una versione
  più recente della stessa famiglia e taglia (es. Granite 4.1 → 4.2), viene nascosto con un rimando.
- **Modelli** — registro di tutti i modelli tracciati, con schede, benchmark e calcolatore VRAM.
- **Classifica** — Intelligence, Coding e Agentic Index di **Artificial Analysis** (via l'endpoint Benchmarks di
  OpenRouter) più le valutazioni proprie di OpenRouter (GPQA, tau-bench), con verdetto GPU per i modelli aperti.
- **Paper** — HF Daily Papers, arXiv per argomento e Alignment Forum, classificati da un LLM per argomento
  (allineamento, interpretabilità meccanicistica, efficienza, retrieval…) e importanza; scegli gli argomenti.
- **Embedding** — leaderboard **MTEB** ufficiale (multilingue, inglese, europeo, codice, documenti lunghi,
  retrieval con ragionamento), punteggio sull'italiano, VRAM, novità e modelli non ancora valutati.
- **Confronto** — fino a 4 modelli fianco a fianco (migliore per riga evidenziato) e un **consiglio LLM**
  sul compito che descrivi: gira nel tuo browser con il tuo account OpenRouter (pulsante "Collega OpenRouter").

## Cosa fa ogni mattina

1. **Raccoglie** da: Hacker News, r/LocalLLaMA, Hugging Face (trending, ~35 organizzazioni ufficiali),
   OpenRouter (nuovi modelli API con prezzi), GitHub (repo nuovi in crescita),
   blog e release (Simon Willison, OpenAI, DeepMind, HF, Ollama, vLLM, Transformers…), paper e MTEB.
2. **Valuta** ogni notizia con un LLM sul tuo profilo (`config.toml`): rilevanza 0-10, categoria,
   titolo e riassunto in italiano, "perché ti interessa".
3. **Riconosce i modelli** citati e li collega a Hugging Face/OpenRouter. Per ognuno:
   - parametri (totali/attivi per i MoE), contesto, licenza, prezzo API, indici Artificial Analysis;
   - **stima VRAM** per ogni quantizzazione (BF16 → IQ2_M) e contesto, contesto massimo, token/s stimati;
   - quantizzazioni GGUF già pronte;
   - **benchmark e caratteristiche** estratti dall'annuncio ufficiale o dalla model card;
   - **versione più recente** della stessa famiglia, se esiste;
   - avvisi utili: repo HF **segnaposto senza pesi**, modelli solo annunciati, repo dichiarati ma non pubblici.
4. Scrive l'**editoriale** del giorno (solo modelli davvero nuovi presentati come novità) e pubblica il sito.

Il calcolo VRAM è deterministico (da `config.json`): gestisce MoE, attenzione ibrida/lineare
(es. Qwen3.8: solo 16 layer su 64 hanno KV cache), sliding window e MLA. Nel sito c'è un
calcolatore interattivo con slider del contesto e tipo di KV cache. Precisione indicativa ±10% sulla memoria,
±25% sui token/s.

## Messa online (10 minuti)

1. **Chiave OpenRouter**: crea un account su openrouter.ai → *Keys* → *Create key*.
   Configurazione attuale: DeepSeek V4.1 Flash per valutare le notizie, Claude Haiku 5.5 per schede,
   benchmark ed editoriale (~0,001 $ a chiamata): **~0,02-0,05 $ al giorno, circa 1 $/mese**, con tetto
   di sicurezza `max_cost_per_run_usd` e i modelli gratuiti come riserva.
   Per restare a 0 € metti `models` e `deep_models` a `["auto:free"]`: senza crediti hai 50 richieste/giorno
   (1000 con ≥10 $ di crediti acquistati), le chiamate durano 10-60 s e a volte i modelli sono congestionati.
2. **Repository GitHub**: crea un repo (pubblico, oppure privato se hai GitHub Pro / Student Pack)
   e fai push di questa cartella.
3. *Settings → Secrets and variables → Actions → New repository secret*:
   `OPENROUTER_API_KEY` = la tua chiave.
4. *Settings → Pages → Build and deployment → Source*: **GitHub Actions**.
5. *Actions → AI Radar giornaliero → Run workflow*. Dopo 3-10 minuti il sito è su
   `https://<utente>.github.io/<repo>/`. Da lì in poi si aggiorna da solo ogni mattina.

Secret opzionali:
- `HF_TOKEN` — token Hugging Face (read) per limiti più alti e config dei modelli gated.

> Il sito su GitHub Pages è raggiungibile da chiunque abbia l'URL (è marcato `noindex`, quindi non finisce sui motori di ricerca).

## Uso in locale

```bash
cp .env.example .env              # e inserisci OPENROUTER_API_KEY
python3 -m radar run              # genera la rassegna di oggi (prima volta: ultimi 7 giorni)
python3 -m radar serve            # apri http://localhost:8000
python3 -m radar check Qwen/Qwen3.8-27B   # verdetto VRAM per qualsiasi repo HF
python3 -m radar run --no-llm     # senza LLM: solo punteggi euristici
```

Nessuna dipendenza: solo Python ≥ 3.11 (libreria standard). Test: `python3 -m unittest discover -s tests -t .`

## Personalizzare

Tutto in `config.toml`:
- `[profile]` — descrizione libera dei tuoi interessi: è quello che l'LLM usa per decidere la rilevanza.
- `[hardware]` — GPU, VRAM, banda, RAM di sistema, contesto di riferimento.
- `[llm]` — budget di chiamate e modelli. `"auto:free"` sceglie da solo i modelli gratuiti disponibili
  (prima il router `openrouter/free`). Con qualche euro puoi mettere un modello economico in testa alla lista.
  Puoi aggiungere altri provider OpenAI-compatibili come riserva.
- `[sources]` — feed RSS, soglie, organizzazioni HF da seguire.

## Struttura

```
radar/            pipeline Python (sources, models, hardware, llm, pipeline)
docs/             sito statico (index.html, app.js, style.css) + docs/data/ generato
state/seen.json   notizie già viste (evita duplicati tra un giorno e l'altro)
.github/workflows/radar.yml   esecuzione giornaliera + deploy su Pages
```

## Note

- Se una fonte non risponde la rassegna prosegue lo stesso; lo stato delle fonti è nel footer del sito.
- GitHub sospende i workflow pianificati dei repo senza attività per 60 giorni: i commit giornalieri del bot
  contano come attività, ma se vedi il workflow disattivato basta riattivarlo da *Actions*.
- I benchmark sono quelli **dichiarati** nelle fonti (estratti dall'LLM con l'istruzione di non inventare):
  ogni tabella riporta il link alla fonte da cui è stata estratta.
