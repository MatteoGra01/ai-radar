# AI Radar

Rassegna quotidiana **personale** su nuovi modelli AI, benchmark, paper e nuove funzionalità, con verdetto
"gira sulla mia RTX 5090?" calcolato dall'architettura reale del modello.
Costo: ~1 $/mese con i modelli economici di OpenRouter (0 € con i soli modelli gratuiti).

## Sezioni del sito

- **Oggi** — editoriale, **le 5 notizie da sapere**, riquadri riassuntivi (nuovi modelli, nuove quantizzazioni,
  paper, tool), nuovi modelli con il tipo di novità (**nuovo modello / nuova versione di… / semplice menzione**),
  **nuove quantizzazioni** comparse per modelli già noti, paper sui tuoi argomenti e la rassegna completa.
  Le **novità dall'ultima visita** sono evidenziate (il browser ricorda cosa hai già visto) e c'è il filtro
  "solo novità". In prima pagina compaiono **solo le ultime versioni**: se di un modello esiste già una versione
  più recente della stessa famiglia e taglia (es. Granite 4.1 → 4.2), viene nascosto con un rimando.
- **Modelli** — registro di tutti i modelli tracciati. Ogni scheda ha: stato di disponibilità (annunciato, repo senza
  pesi, pesi gated, pesi aperti, quantizzazioni pronte), categoria GPU (**senza quantizzare / quantizzato /
  GPU + RAM con offload / non supportato**), requisiti **dichiarati dal produttore** separati dalla stima,
  **"Quando sceglierlo"** (punti di forza e limiti ricavati dai dati, alternative nella stessa fascia),
  calcolatore VRAM con la metodologia spiegata, **comandi pronti** per llama.cpp / Ollama / vLLM e
  **"Qualità dei dati"** (da dove viene ogni informazione, avvisi, data dell'ultima verifica).
- **Classifica** — Intelligence, Coding e Agentic Index di **Artificial Analysis** (via l'endpoint Benchmarks di
  OpenRouter) più le valutazioni proprie di OpenRouter (GPQA, tau-bench), con verdetto GPU per i modelli aperti.
  «n.v.» = indice non misurato (non un punteggio basso).
- **Paper** — HF Daily Papers, arXiv per argomento e Alignment Forum, classificati da un LLM per argomento
  (allineamento, interpretabilità meccanicistica, efficienza, retrieval…) con **due voti**: rilevanza per te e
  valore scientifico nel suo campo. **Cliccando un paper** si apre un'overview semplice (problema, idea, risultati,
  limiti, parole chiave) e l'abstract originale. Date distinte (pubblicazione, comparsa su HF, rilevamento),
  "salva per dopo" e "letto".
- **Embedding** — leaderboard **MTEB** ufficiale (multilingue, inglese, europeo, codice, documenti lunghi,
  retrieval con ragionamento): scegli il **task che ti serve** (Retrieval, Classification, Pair Classification, STS…)
  e la classifica si ordina su quello; filtri per lingua, VRAM e dimensione; posizione nella vista e rank MTEB.
- **Confronto** — fino a 4 modelli fianco a fianco (migliore per riga evidenziato), tabella **"Quale sceglieresti
  per…"** (coding in locale, massima qualità, agenti, API economica, contesto lungo, velocità) calcolata sui dati,
  e un **consiglio LLM** sul compito che descrivi: gira nel tuo browser con il tuo account OpenRouter.
- **Archivio** — tutti i giorni passati, con ricerca in tutte le notizie.

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
4. **Controlla la qualità delle schede** prima di pubblicarle: unisce i duplicati (un "annunciato" che poi è un repo
   già noto), ricontrolla ogni settimana su Hugging Face lo stato dei repository (esiste, gated, contiene i pesi),
   riconosce i repo ufficiali anche quando il nome dell'organizzazione differisce (VillanovaAI → Villanova2) e
   annota gli avvisi sui dati mancanti.
5. Scrive l'**overview** dei paper più rilevanti (punteggio ≥ 6, max 60 al giorno).
6. Scrive l'**editoriale** del giorno (solo modelli davvero nuovi presentati come novità) e pubblica il sito.

Il calcolo VRAM è deterministico (da `config.json`): gestisce MoE, attenzione ibrida/lineare
(es. Qwen3.8: solo 16 layer su 64 hanno KV cache), sliding window e MLA. Se il modello non entra in VRAM ma
entra in VRAM + RAM, il verdetto è **GPU + RAM** con la velocità stimata dalla banda della RAM (con i MoE gli
esperti vanno in RAM con `llama.cpp --n-cpu-moe`). Senza `config.json` leggibile usa i parametri attivi
dichiarati dal produttore. Nel sito c'è un calcolatore interattivo con slider del contesto e tipo di KV cache.
Sono **stime**, non misure: precisione indicativa ±10% sulla memoria, ±25% sui token/s (batch 1, contesto breve).

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
python3 -m radar papers --reclassify   # dopo aver cambiato gli argomenti in config.toml: riclassifica l'archivio
python3 -m radar papers --overviews    # genera le overview mancanti dei paper più rilevanti
python3 -m radar validate              # controllo qualità delle schede (senza LLM)
python3 -m radar deep <id-modello>     # rifà l'analisi approfondita di un modello
```

Nessuna dipendenza: solo Python ≥ 3.11 (libreria standard). Test: `python3 -m unittest discover -s tests -t .`

## Personalizzare

Tutto in `config.toml`:
- `[profile]` — descrizione libera dei tuoi interessi: è quello che l'LLM usa per decidere la rilevanza.
- `[hardware]` — GPU, VRAM, banda, RAM di sistema (e quanta lasciarne al sistema, banda della RAM per l'offload),
  contesto di riferimento.
- `[llm]` — budget di chiamate e modelli. `"auto:free"` sceglie da solo i modelli gratuiti disponibili
  (prima il router `openrouter/free`). Con qualche euro puoi mettere un modello economico in testa alla lista.
  Puoi aggiungere altri provider OpenAI-compatibili come riserva.
- `[sources]` — feed RSS, soglie, organizzazioni HF da seguire.
- `[papers]` — argomenti, fonti, soglia e numero massimo di overview al giorno.

## Struttura

```
radar/            pipeline Python (sources, models, hardware, llm, pipeline)
radar/validate.py controllo qualità delle schede modello
docs/             sito statico (index.html, app.js, style.css) + docs/data/ generato
                  (paper_details.json = abstract e overview, caricato solo quando apri un paper)
state/seen.json   notizie già viste (evita duplicati tra un giorno e l'altro)
.github/workflows/radar.yml   esecuzione giornaliera + deploy su Pages
```

## Note

- Se una fonte non risponde la rassegna prosegue lo stesso; lo stato delle fonti è nel footer del sito.
- GitHub non garantisce l'orario dei workflow pianificati (possono slittare anche di un'ora o saltare): per questo
  oltre all'orario principale (06:23 in inverno) ci sono due riserve (07:47 e 10:17), che non fanno nulla se la
  rassegna del giorno esiste già. Puoi sempre lanciarla a mano da *Actions → Run workflow*.
  Sui repo nuovi i cron di GitHub possono saltare per giorni interi: per un avvio affidabile un pianificatore esterno
  può lanciare il workflow (`workflow_dispatch`) con `automatico=true`, che come le riserve non fa nulla se la
  rassegna del giorno esiste già.
- GitHub sospende i workflow pianificati dei repo senza attività per 60 giorni: i commit giornalieri del bot
  contano come attività, ma se vedi il workflow disattivato basta riattivarlo da *Actions*.
- I benchmark sono quelli **dichiarati** nelle fonti (estratti dall'LLM con l'istruzione di non inventare):
  ogni tabella riporta il link alla fonte da cui è stata estratta.
