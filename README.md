# Explainable Agent (Local LM Studio)

Local LLM ile calisan, arac kullanimi ve karar izini raporlayan aciklanabilir agent kutuphanesi.

Bu repo iki hedefe odaklanir:
1. Uretim benzeri, izlenebilir agent calistirma akisi.
2. Tool-calling kalitesini dataset ile olcen eval hatti.

## Ozellikler

- Adim adim karar izi: `action`, `confidence`, `rationale`, `evidence`
- Tool cagrisi ve sonuc kaydi
- Faithfulness metrikleri:
  - alternatif cevap benzerligi
  - tool-support skoru
- Cikti artefact'lari:
  - `trace.json` (kisa)
  - `trace_full.json` (detayli)
  - `report.md`
- Tool-calling eval:
  - random/head sampling
  - parse/repair/guard istatistikleri
  - arguman hata dagilimi

## Proje Yapisi

- `explainable_agent/`: kutuphane kodu
- `scripts/`: eval ve smoke-test scriptleri
- `tests/`: birim testler
- `data/evals/`: ornek ve benchmark datasetleri

## Kurulum

### Secenek A: Gelistirme (requirements)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Secenek B: Paket olarak (pyproject)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

`.env.example` dosyasini kopyalayip kendi ortam degiskenlerini ayarlayabilirsin.

## Hemen Basla

LM Studio:
1. Local server ac.
2. Bir model yukle (ornek: `gpt-oss-20b`).
3. Endpoint varsayilan: `http://localhost:1234/v1`

Model listesini kontrol et:

```bash
python -m explainable_agent.cli --list-models
```

Agent calistir:

```bash
python -m explainable_agent.cli --model gpt-oss-20b --reasoning-effort high --task "calculate_math: (215*4)-12"
```

Konsol script ile:

```bash
explainable-agent --model gpt-oss-20b --task "sqlite_init_demo"
```

## Yerlesik Tool'lar

- `calculate_math`
- `read_text_file`
- `list_workspace_files`
- `now_utc`
- `sqlite_init_demo`
- `sqlite_list_tables`
- `sqlite_describe_table`
- `sqlite_query`
- `sqlite_execute`

SQLite ornek:

```bash
python -m explainable_agent.cli --sqlite-db data/demo.db --task "sqlite_init_demo"
python -m explainable_agent.cli --sqlite-db data/demo.db --task "sqlite_query: SELECT name, city FROM customers ORDER BY id;"
```

## Eval Kullanim

Mini eval:

```bash
python scripts/eval_hf_tool_calls.py --dataset data/evals/hf_xlam_fc_sample.jsonl --model gpt-oss-20b --reasoning-effort high --limit 10
```

ComplexFuncBench subset:

```bash
python scripts/eval_hf_tool_calls.py --dataset data/evals/hf_complexfuncbench_first_turn_100.jsonl --model gpt-oss-20b --reasoning-effort high --limit 10 --sampling random
```

BFCL SQL:

```bash
python scripts/eval_hf_tool_calls.py --dataset data/evals/bfcl_sql/BFCL_v3_sql.json --model gpt-oss-20b --reasoning-effort high
```

Eval ciktilari:

- `runs/evals/hf_tool_eval_<timestamp>/summary.json`
- `runs/evals/hf_tool_eval_<timestamp>/details.json`
- `runs/evals/hf_tool_eval_<timestamp>/report.md`

## Prepublish Check

GitHub'a push etmeden once:

```bash
python scripts/prepublish_check.py
```

Bu script derleme ve cekirdek testleri calistirir.

## Konfigurasyon

- `LMSTUDIO_BASE_URL` (default: `http://localhost:1234/v1`)
- `LMSTUDIO_API_KEY` (default: `lm-studio`)
- `AGENT_MODEL` (default: `gpt-oss-20b`)
- `AGENT_REASONING_EFFORT` (default: `high`)
- `AGENT_MAX_STEPS` (default: `6`)
- `AGENT_RUNS_DIR` (default: `runs`)
- `AGENT_WORKSPACE` (default: `.`)
- `AGENT_TEMPERATURE` (default: `0.2`)
- `AGENT_SQLITE_DB` (default: `data/agent.db`)

## Portfoy Notu

Bu repo su an `v0.1.0` deneysel kutuphane seviyesindedir:
- Yerel model + explainability + eval bir arada
- Tekrar edilebilir benchmark akisi
- Arac guvenlik kontrolleri (path/SQL guard)
- Mimari ozeti: `docs/ARCHITECTURE.md`

## Lisans

MIT
