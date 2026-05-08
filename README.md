# MindLog — Extraction Pipeline Preliminary Validation

> **Reference:** MindLog Research Proposal §7.4 — *Preliminary Validation: Extraction Pipeline Accuracy*

Validates the structured-indicator extraction pipeline against the
[NLP Mental Health Conversations](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations)
dataset from Kaggle, producing the evaluation table reported in the proposal.

---

## How to Run

### Prerequisites

- Python 3.10+
- An OpenAI API key with GPT-4o access
- The Kaggle dataset CSV file

### Step-by-step Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env .env
# Open .env in any editor and paste your key:
#   OPENAI_API_KEY=sk-proj-...

# 3. Download the dataset
#    Go to: https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations
#    Download "combined_data.csv" and place it here:
#      00_data/00_raw/combined_data.csv

# 4. Run the full pipeline
python 02_scripts/01_run_validation.py
```

The full run makes ~100 GPT-4o API calls (50 annotation + 50 extraction).
Expect **5–10 minutes** and roughly **$1–3 in API cost** depending on
conversation lengths.

### CLI Options

```bash
# Re-evaluate without re-calling the API (uses cached JSON files)
python 02_scripts/01_run_validation.py --evaluate-only

# Re-run extraction only (keep existing ground truth annotations)
python 02_scripts/01_run_validation.py --skip-annotation

# Re-run annotation only (keep existing extraction results)
python 02_scripts/01_run_validation.py --skip-extraction

# Override the sample size (default: 50)
python 02_scripts/01_run_validation.py --sample-size 100
```

### Output Files

After the run completes, results are in:

| File | Contents |
|:-----|:---------|
| `00_data/01_processed/sampled_conversations.json` | The N sampled conversations used as input |
| `00_data/01_processed/ground_truth_labels.json` | CoT-annotated ground truth labels + reasoning |
| `00_data/01_processed/extraction_results.json` | Pipeline-extracted predicted labels |
| `00_data/02_results/evaluation_metrics.csv` | **Table 1** for your proposal — copy-paste ready |
| `00_data/02_results/evaluation_report.json` | Full report: confusion matrices, per-class breakdowns, bootstrap CIs, error analysis |

The terminal also prints the formatted Table 1 during Step 4.

---

## How It Works — Logic Walkthrough

### The Problem This Solves

MindLog's core value chain is:

```
Daily conversation → Structured indicators → Weekly summary → Pre-visit briefing
```

This validation answers: **"How accurately does Step 1→2 actually work?"**
If the extraction pipeline can't reliably turn messy conversational text into
structured fields (mood, energy, sleep, themes, risk), then everything
downstream — summaries, briefings, trend charts — is unreliable.

### Pipeline Architecture (5 Steps)

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Load & Sample                                           │
│   combined_data.csv → filter by length → random N=50 sample     │
│   Output: sampled_conversations.json                            │
├─────────────────────────────────────────────────────────────────┤
│ Step 2: Annotate (Ground Truth)                                 │
│   Each sample → GPT-4o with CoT prompt → reasoned labels        │
│   "Why is this negative? Because the user says '...'"           │
│   Output: ground_truth_labels.json                              │
├─────────────────────────────────────────────────────────────────┤
│ Step 3: Extract (Pipeline Under Test)                           │
│   Each sample → GPT-4o with extraction prompt → predicted labels│
│   This is what MindLog would run in production, single-pass     │
│   Output: extraction_results.json                               │
├─────────────────────────────────────────────────────────────────┤
│ Step 4: Evaluate                                                │
│   Compare Step 3 predictions vs. Step 2 ground truth            │
│   Per-field: accuracy, precision, recall, F1, Cohen's kappa     │
│   Aggregate: macro averages, exact-match rate, bootstrap CIs    │
├─────────────────────────────────────────────────────────────────┤
│ Step 5: Export                                                  │
│   evaluation_report.json (full) + evaluation_metrics.csv (Table)│
└─────────────────────────────────────────────────────────────────┘
```

### Step 1 — Load & Sample (`01_src/01_data/loader.py`)

The Kaggle dataset has two columns: **Context** (what the user said) and
**Response** (what the counselor replied). We only extract from Context —
the user's own words — because that's what MindLog's daily log captures.

The loader filters out entries that are too short (<80 chars, not enough
signal) or too long (>3000 chars, unrealistic for a 5-10 min daily log).
Then it draws a reproducible random sample (seed=42, N=50).

Why N=50? The proposal's original benchmark used N=20 synthetic conversations.
50 real conversations from an external dataset is a stronger validation while
staying within reasonable API cost for a preliminary study.

### Step 2 — Annotate (`01_src/02_extraction/extractor.py`)

This is the **ground truth generation** step. The key design question is:
*"If we're using GPT-4o to extract AND to annotate, aren't we just comparing
the model against itself?"*

The answer is that the two prompts are deliberately different:

| | Extraction (Step 3) | Annotation (Step 2) |
|:---|:---|:---|
| **Goal** | Fast, cheap production extraction | High-fidelity gold-standard labels |
| **Prompt** | "Return JSON with these 5 fields" | "For each field, explain your reasoning citing specific phrases, THEN label" |
| **CoT** | No — direct labeling | Yes — forced Chain-of-Thought per field |
| **Max tokens** | 512 | 1024 |
| **Output** | `{field: label}` | `{reasoning: {...}, labels: {...}}` |

The CoT annotation prompt forces the model to justify each label by pointing
to specific text evidence *before* committing to a label. This reduces
arbitrary or sloppy labeling. When the extraction prompt (which skips
reasoning) disagrees with the annotation prompt, that disagreement is
a genuine signal about extraction difficulty — not just random noise.

The reasoning is also saved, which lets you manually audit any label you're
uncertain about.

### Step 3 — Extract (`01_src/02_extraction/extractor.py`)

This simulates MindLog's production extraction module. For each conversation:

1. Send the user's text to GPT-4o with the extraction system prompt
2. Force JSON-mode output (`response_format: json_object`)
3. Parse and validate that all 5 fields have legal label values
4. Retry up to 3 times on failure (rate limits, malformed JSON)

The 5 extraction fields map directly to Proposal §4.2:

| Field | What it captures | Why it matters for MindLog |
|:------|:-----------------|:--------------------------|
| **Affect Valence** | positive / neutral / negative | Core mood tracking — feeds the trend chart |
| **Energy Level** | low / medium / high | Behavioral activation mission difficulty |
| **Sleep Quality** | poor / fair / good / not_mentioned | Clinician briefing; common depression indicator |
| **Dominant Theme** | emotional / relationships / work / ... | Weekly summary theme analysis |
| **Risk Indicators** | none / low / moderate / high | Safe-messaging protocol trigger |

### Step 4 — Evaluate (`01_src/03_evaluation/evaluator.py`)

For each of the 5 fields, we compute:

- **Accuracy** — what fraction did the extractor get right?
- **Precision (macro)** — when it predicts "negative", how often is that correct? Averaged across all label categories.
- **Recall (macro)** — of all truly "negative" samples, how many did it catch? Averaged across all label categories.
- **F1 (macro)** — harmonic mean of precision and recall.
- **Cohen's Kappa** — agreement adjusted for chance. Accuracy can be high even if both raters just pick the most common label. Kappa corrects for that.

Then two aggregate metrics:

- **Overall (macro avg)** — average of all 5 fields' accuracy/precision/recall/F1.
- **Exact Match** — what fraction of samples have ALL 5 fields correct simultaneously. This is the hardest metric: even 90% per-field accuracy gives ~0.9⁵ ≈ 59% exact match.

Finally, **bootstrap confidence intervals** (1000 resamples) for each field's
accuracy, so you can report "Affect Valence accuracy: 0.88 [0.78, 0.96]"
with proper uncertainty quantification.

### Step 5 — Export

Writes two files:
- `evaluation_metrics.csv` — the Table 1 you can paste directly into your proposal
- `evaluation_report.json` — the complete report including confusion matrices,
  per-class precision/recall, the top misclassification patterns (e.g.
  "neutral → negative ×4"), and which specific samples had the most errors

---

## Extraction Fields

| Field             | Labels                                                    |
|:------------------|:----------------------------------------------------------|
| Affect Valence    | positive · neutral · negative                             |
| Energy Level      | low · medium · high                                       |
| Sleep Quality     | poor · fair · good · not_mentioned                        |
| Dominant Theme    | emotional · relationships · work_academic · physical_health · existential · daily_routine · other |
| Risk Indicators   | none · low · moderate · high                              |

---

## Project Structure

```
mindlog_validation/
├── 00_data/
│   ├── 00_raw/                  # Kaggle CSV (you download this)
│   ├── 01_processed/            # Sampled conversations, annotations, extractions
│   └── 02_results/              # Evaluation report + metrics CSV
├── 01_src/
│   ├── 00_common/               # Config loader, logger
│   │   ├── config_loader.py     # Loads config.yaml + .env
│   │   └── logger.py            # Shared logging format
│   ├── 01_data/
│   │   └── loader.py            # CSV ingestion, filtering, sampling
│   ├── 02_extraction/
│   │   └── extractor.py         # LLM prompts, extraction, annotation
│   └── 03_evaluation/
│       └── evaluator.py         # Metrics, confusion matrix, Table 1
├── 02_scripts/
│   └── 01_run_validation.py     # Main pipeline orchestrator
├── 03_configs/
│   └── config.yaml              # All parameters in one place
├── 04_artifacts/                # (reserved for future model artifacts)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Dataset

**NLP Mental Health Conversations**
— [kaggle.com/datasets/thedevastator/nlp-mental-health-conversations](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations)

| Property | Details |
|:---------|:--------|
| **File** | `combined_data.csv` |
| **Columns** | `Context` (user utterance), `Response` (counselor reply) |
| **Domain** | Mental health counseling conversations |
| **Language** | English |
| **Tags** | NLP · Psychology · Mental Health · Text Mining |

The dataset contains paired mental health conversation exchanges where the
**Context** column captures a user's open-ended self-disclosure — describing
their emotional state, stressors, relationships, sleep patterns, daily
experiences, or crisis situations in their own unstructured language — and the
**Response** column contains a counselor's or support agent's reply.

### Why this dataset fits MindLog

MindLog's extraction pipeline must parse **naturalistic, unstructured
self-report text** — the kind of thing a user types during a 5-10 minute
evening check-in on the bus home. This dataset's Context column exhibits
exactly those patterns:

- **Affect expression** — "I've been feeling really down lately", "I'm anxious
  all the time"
- **Theme mention** — work stress, relationship conflicts, health concerns
- **Energy/motivation cues** — "I can't get out of bed", "I've been pushing
  myself to stay active"
- **Sleep references** — "I haven't slept well in weeks", "I keep waking up
  at 3am"
- **Risk language** — varying from vague hopelessness to explicit crisis
  statements

The conversations span a range of mental health topics (depression, anxiety,
trauma, relationships, existential concerns) and severity levels, providing
natural variation across all five extraction fields.

### How the pipeline uses it

Only the **Context** column (user's words) is fed to the extraction pipeline.
The Response column is retained in the sampled data for reference but is not
part of the extraction input — consistent with MindLog's design where the
chatbot's replies are generated separately and the extraction module operates
solely on what the user said.

The loader filters conversations by length (80–3000 characters) to approximate
realistic daily log inputs: long enough to contain extractable signal, short
enough to match the expected 5-10 minute session length.

---

## Customization

All parameters live in `03_configs/config.yaml`. Key things you might change:

- `dataset.sample_size` — increase for a stronger benchmark (costs more API)
- `dataset.min_context_length` / `max_context_length` — adjust filtering
- `llm.model` — swap to `gpt-4o-mini` for cheaper runs during development
- `evaluation.bootstrap_iterations` — increase for tighter CIs
- `extraction.fields` — add or modify extraction fields

---

## License

Research prototype — not for clinical use.
