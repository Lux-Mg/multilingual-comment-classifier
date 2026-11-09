# Multilingual Comment Classifier

A small inference service that classifies movie comments by **sentiment** (positive/negative/neutral) and detects their **language** (en/es/ru/other), using a local LLM via Ollama and a FastAPI wrapper. Built as part of my AI/ML portfolio.

The architecture is straightforward. The main part is the benchmark and the A/B test of prompt language, both documented below.

## What it does

```bash
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Фильм нормальный, но ничего особенного"}'

# → {"sentiment": "neutral", "language": "ru"}
```

The service uses **Qwen 2.5 7B** through Ollama (running on the host with GPU) and forces a JSON schema via Ollama's `format` parameter, so the response is always a valid object with the expected fields.

## Results

The benchmark sample comes from my thesis project, LenguaTrends. The source corpus is not tracked in this repository; the two committed result CSVs preserve the same 100 sampled comments and the outputs from an existing **XLM-RoBERTa-based** pipeline (`cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`).

I benchmarked the local LLM against this baseline on a sample of 100 comments (`random_state=42`):

| Metric | XLM-R baseline (reference) | Qwen 2.5 7B (English prompt) |
|---|---|---|
| Sentiment agreement | n/a | **67.8%** |
| Language agreement | n/a | **97.7%** |
| Coverage of "unsupported" comments | 0/13 | **12/13 (92.3%)** |
| Latency P50 | (negligible, runs on CPU) | ~1.0 s |
| Latency P95 | n/a | ~1.32 s |

Numbers from `analyze_benchmark.py` over `data/benchmark_results_en.csv`. Agreement is computed on the 87 comments the baseline could classify; the other 13 are `unsupported`.

A few things worth pointing out:

- **The "rescue" metric measures coverage, not correctness.** The earlier pipeline relies on `langdetect` as a pre-processing step. When it fails, a comment is marked `language="unsupported"`. In the English-prompt run, Qwen assigned one of the three target languages to 12 of those 13 comments; without manual ground truth, this should not be read as 92.3% language accuracy.
- **Sentiment agreement is not the same as sentiment accuracy.** Without a manually labeled gold standard, I can only compare against XLM-R, which is itself imperfect. The 67.8% means the two models disagree on roughly one in three borderline cases. Some of those disagreements I'd argue the LLM gets right (e.g. sarcasm: *"Can't wait for Avatar 9: moss and pollen"*: XLM-R said positive, LLM said negative).
- **Latency is a function of running on a 2060 Super.** On a real GPU you'd see lower numbers; on CPU-only it's roughly 30–60× slower. The point of the service isn't competitive throughput, it's showing that a local LLM can be a viable replacement for a specialized pipeline when the operational constraints allow it.

## A/B test: prompt language matters

The first version of the classifier used a prompt written in Spanish. Once I had the benchmark running, I noticed an odd pattern in the language distribution:

| Language | XLM-R reported | Qwen, Spanish prompt | Qwen, English prompt |
|---|---|---|---|
| es | 33 | **50** | 39 |
| en | 28 | 22 | 31 |
| ru | 26 | 28 | 28 |
| unsupported / other | 13 | 0 | 2 |

The Spanish prompt was biasing the model toward Spanish, by 17 cases, which is a lot for a 100-sample test. Looking at individual disagreements, clearly-English comments (*"Whaaat? A new Raimi movie?! Big fan."*, *"Gonna tell my kids this is Half Nelson"*) were being labeled `es` when the prompt was in Spanish.

Switching the prompt to English fixed most of it:

| | Spanish prompt | English prompt | Δ |
|---|---|---|---|
| Sentiment agreement | 63.2% | 67.8% | +4.6 pp |
| Language agreement | 92.0% | 97.7% | +5.7 pp |

The prompt is parameterized (`prompt_lang='en'` is the default), so both versions are reproducible:

```python
from clasificar import clasificar_comentario

clasificar_comentario("This was awful", prompt_lang='es')  # biased
clasificar_comentario("This was awful", prompt_lang='en')  # default, less biased
```

**Lesson:** the language a prompt is *written in* leaks into the output, even when the prompt explicitly asks for something else. The Ollama `format` parameter guarantees structure, not correctness. Worth keeping in mind for any classifier where one of the output fields is a language code.

## Architecture

Two processes on the same machine:

```
┌────────────────────────────────────────────────────────────┐
│ Host (WSL2 on Windows, NVIDIA RTX 2060 Super, 8 GB VRAM)   │
│                                                            │
│   ┌──────────────────────┐      ┌────────────────────┐     │
│   │  Ollama service      │◄─────│  Docker container  │     │
│   │  (systemd, GPU)      │      │  (FastAPI app)     │     │
│   │  Port 11434          │      │  Port 8000 → 8001  │     │
│   │  Qwen 2.5 7B (4.7GB) │      │                    │     │
│   └──────────────────────┘      └────────────────────┘     │
│              ▲                            │                │
│              └────────────────────────────┘                │
│           http://host.docker.internal:11434                │
└────────────────────────────────────────────────────────────┘
                              ▲
                              │
                       client (curl / browser)
                       http://localhost:8001
```

I keep Ollama on the host (not inside Docker) for two reasons:

1. **GPU access.** Passing a CUDA device into a container in WSL2 is possible but adds a layer of fragility. The host has working CUDA out of the box; reusing it is simpler.
2. **Cold start cost.** The 4.7 GB model has to be downloaded once. If Ollama ran inside the container, that download (and the VRAM load) would happen every time the image is rebuilt.

The container talks to the host via `host.docker.internal`, which Docker Desktop maps to the host's IP. The `OLLAMA_HOST` env var (set in `docker-compose.yml`) tells the `ollama` Python client to point there instead of `localhost:11434` (which from inside the container would mean the container itself).

## Stack

- **Python 3.12**: application code
- **Ollama**: local LLM runtime (host)
- **Qwen 2.5 7B**: open-weights model with strong multilingual support, particularly for Russian
- **FastAPI + Uvicorn**: API layer, picked over Flask for built-in validation via Pydantic and automatic OpenAPI docs
- **pandas**: data handling for the benchmark scripts
- **Docker + docker-compose**: container packaging and host-network bridging

## Running it

You need Ollama installed on the host, with the model pulled:

```bash
# Install Ollama (Linux/WSL):  https://ollama.com/download
ollama pull qwen2.5:7b
```

Then the container:

```bash
git clone https://github.com/Lux-Mg/multilingual-comment-classifier.git
cd multilingual-comment-classifier
docker compose up --build
```

The API will be available at `http://localhost:8001`. Swagger UI is at `http://localhost:8001/docs`.

For development without Docker:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api:app --reload --port 8000
```

## Repository layout

```
.
├── api.py                  # FastAPI app, single endpoint /classify
├── clasificar.py           # Classification function, prompt_lang parameter
├── procesar_csv.py         # Exploratory: load CSV, show a sample
├── benchmark.py            # Runs N classifications, saves results to CSV
├── analyze_benchmark.py    # Metrics from a single benchmark CSV
├── compare_prompts.py      # A/B comparison of two benchmark CSVs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/
│   ├── comentarios.csv     # (not in git, comes from my thesis project)
│   ├── benchmark_results_es.csv
│   └── benchmark_results_en.csv
└── README.md
```

`data/comentarios.csv` is excluded from version control (it contains real user comments from YouTube and would inflate the repo). The benchmark result CSVs are kept for reproducibility of the A/B comparison.

## Limitations

A short list of things I'd improve given more time:

- **No labeled ground truth.** The "accuracy" numbers compare against XLM-RoBERTa, which is itself imperfect. A proper evaluation would require ~200 comments labeled by hand. The disagreement analysis suggests some of the gap is the LLM being *more* correct (sarcasm, code-switching), not less.
- **Single LLM tested.** Llama 3.1 8B and Qwen 3 would be obvious next comparisons. Qwen 2.5 was chosen for its multilingual coverage, but it's not the only option.
- **No public deployment.** The repository packages the API as a container for local use, but it does not provide or claim a production deployment.
- **No batching or async.** Each request is a separate Ollama call. For sustained traffic you'd want either Ollama's built-in batch support or a queue in front of it.
- **The `'other'` language code is coarse.** Languages outside en/es/ru collapse to a single bucket. Expanding the enum would require either more training data for the validator or just trusting the model's free-text output.

## What I learned

In rough order of usefulness:

1. **Local LLMs can cover cases that pre-processing pipelines drop.** In this sample, the English-prompt run assigned a target language to 12 of 13 comments marked `unsupported` by the earlier pipeline; manual labels would be needed to measure correctness.
2. **Prompt language is a confounder.** It's the kind of bias you don't notice until you measure it. The Spanish prompt was off by 17/100 on language detection compared to the English one.
3. **Structured output ≠ correct output.** Ollama's `format` parameter guarantees the JSON is well-formed and the values are within the enum, but the model still picks the wrong value sometimes. Schema validation gives you safety, not accuracy.
4. **`host.docker.internal` is the cleanest way to bridge a containerized app to a host service in WSL2.** With `extra_hosts: host-gateway` in compose, it just works.
5. **`no_proxy` is non-negotiable when running a system VPN.** Without it, my local proxy intercepted localhost traffic and turned every API call into a silent 502.

---

Built by Luis Mendoza · [GitHub](https://github.com/Lux-Mg) · 2026
