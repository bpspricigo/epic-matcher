# Epic Matcher

Automatically route unclassified bug tickets to the right engineering epic using a trained machine learning model — then let a human review and confirm each one.

Built for teams that receive a steady stream of defect tickets that need to be triaged into topic areas before engineers can pick them up. The model does the first pass; you do the last mile.

**~74% accuracy out of the box** · 5-fold cross-validation · falls back to keyword matching without the model

---

## The problem it solves

When a bug queue grows faster than a triage team can read it, tickets pile up unassigned. The right engineer never sees them, the wrong team gets pinged, and backlog hygiene breaks down.

Epic Matcher was built for a navigation software team dealing with exactly this: hundreds of defect tickets per sprint, each needing to be routed to one of ~25 engineering epics (map rendering, routing, positioning, speech, EV charging, etc.). The tickets were written in a mix of English and German, contained hex error codes, GPS coordinates, and firmware version strings, and followed a structured "Expected / Observed result" template — but not always consistently.

The tool fetches all tickets with no Epic Link, ranks predictions by confidence, and lets a reviewer step through them interactively — accepting the suggestion or picking a different epic. Every decision is logged so you can measure how often the model was right.

---

## How it works

```
Jira (unclassified tickets)
        │
        ▼
  Text preprocessing
  ├─ Extract "Expected / Observed" sections  ← the signal-rich part of defect reports
  ├─ Strip Jira markup, normalize noise      ← version strings, hex codes, coordinates
  └─ Lemmatize (EN + DE via spaCy)           ← bilingual support, optional
        │
        ▼
  3-branch TF-IDF feature union
  ├─ Ticket title        → word n-grams (1–2), lemmatized
  ├─ EO description      → word n-grams (1–2), lemmatized
  └─ EO description      → char n-grams (3–5), raw        ← typo-robust
        │
        ▼
  LinearSVC classifier
  → ranked predictions with confidence scores
        │
        ▼
  Interactive review (optional)
  → accept · pick different · skip → write Epic Link to Jira → append to audit log
```

**Why LinearSVC and not a transformer?** Dense NN, Conv1D, and BiLSTM were all tested. They scored 21–59% — worse than TF-IDF. With ~6,700 training samples spread across 25+ imbalanced classes and highly domain-specific vocabulary, sparse TF-IDF features outperform dense embeddings. The char n-gram branch also handles the typos and mixed-language tokens that embeddings struggle with.

The key preprocessing insight: defect reports contain a lot of noise (stack traces, table markup, step-by-step reproduction instructions) but the **Expected Result / Observed Result** sections are dense with the actual signal. Extracting just those sections — 500 chars max — and giving them their own model branch was the single biggest accuracy improvement.

---

## Adapting to your domain

This tool is domain-agnostic. You need:

1. A Jira project with tickets that should be classified into epics
2. Existing labeled tickets to train on (tickets that already have an Epic Link set)
3. A `config.yaml` pointing at your instance

The navigation example used ~25 epics and ~6,700 labeled issues. Accuracy will scale with how many labeled examples you have per class and how distinct the vocabulary is between topics. Highly overlapping epics (e.g. two "guiding" epics that were split for organizational reasons) will confuse the model just as they'd confuse a human.

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional but recommended — improves accuracy ~2-3%
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### 2. Configure

Copy the default config and fill in your values:

```bash
cp config.default.yaml config.yaml
```

`config.yaml` is gitignored so your URLs and credentials stay local. Key fields:

```yaml
jira:
  primary_url: "https://jira.yourcompany.com"
  primary_host: "jira.yourcompany.com"   # must match your .netrc entry
  default_filter: "My Team - Unclassified Bugs"
  issue_types: ["Bug", "Defect"]

epics:
  PROJ-1001: "Bug - Authentication"
  PROJ-1002: "Bug - Data Sync"
  PROJ-1003: "Bug - UI / Rendering"
  # ...
```

### 3. Set up authentication

Epic Matcher reads your API token from `.netrc` (Unix) or `_netrc` (Windows):

```
machine jira.yourcompany.com
login your-username
password your-api-token
```

### 4. Run

```bash
# Fetch & classify — prints a ranked table, no writes
python3 epic_matcher.py --jira

# Interactive review — step through each ticket, confirm or override, writes to Jira
python3 epic_matcher.py --jira --interactively
```

Set `CHANGE_JIRA_DATA = False` in `epic_matcher.py` for a full dry-run (reads only).

---

## Usage reference

### Batch classification (read-only)

```bash
python3 epic_matcher.py --jira
python3 epic_matcher.py --jira --max 50
python3 epic_matcher.py --jira --jql '"Epic Link" is EMPTY AND project = MYPROJ'
```

Output is a table sorted by predicted epic, showing the top-3 ranked predictions and their scores for each ticket:

```
#    Ticket                 Title                                            | Epics in order
1    PROJ-9142              GPS position jumps 200m after tunnel exit        | 1  Bug - Positioning (+1.82) | 2  Bug - Map (+0.41) | ...
2    PROJ-9138              Voice guidance silent on highway merge            | 1  Bug - Speech    (+2.11) | 2  Bug - Guiding  (+0.93) | ...
```

### Interactive review

The main workflow. For each ticket the tool shows the title, a snippet of the Expected/Observed description, the top prediction and its score, then asks what to do:

```
════════════════════════════════════════════════════════════════════
  Ticket 3 of 47  |  45 ticket(s) left (including this one)
════════════════════════════════════════════════════════════════════
  Key   : PROJ-9151
  Title : Route recalculation loops after missed turn

  Description:
    Expected: new route calculated within 3s of missed turn.
    Observed: app recalculates continuously, route never stabilises.

  Suggested Epic : Bug - Routing  (PROJ-1004)
  Score          : +1.74

  What do you want to do?
  ❯ Accept suggested epic
    Choose a different epic
    Skip this ticket
```

Accepting writes the Epic Link directly to Jira and appends a row to `epic_choices_log.csv`.

### Focus on a specific epic

Useful when you want to clear a particular backlog area first:

```bash
# See all tickets predicted for "Bug - Routing", then the rest
python3 epic_matcher.py --jira --epic "Bug - Routing"

# Same, but step through interactively starting with that group
python3 epic_matcher.py --jira --epic "Bug - Routing" --interactively
```

### Classify a single ticket without Jira

```bash
python3 epic_matcher.py \
  "Route not updating after road closure" \
  "Expected: reroute within 10s. Observed: original route kept, no reroute triggered."
```

### Programmatic API

```python
from epic_matcher import classify_ticket

epic_key, epic_name, score, alternatives = classify_ticket(
    title="Map tiles not loading at zoom level 15",
    description="Expected: tiles render within 2s. Observed: blank tiles shown indefinitely."
)

print(f"Predicted: {epic_name}  (score: {score:+.2f})")
for alt in alternatives:
    print(f"  {alt['score']:+.2f}  {alt['epic_name']}")
```

---

## Training your own model

You need existing labeled tickets — i.e. tickets that already have an Epic Link set. The more per class, the better; aim for at least 100 per epic.

**Step 1 — Fetch training data from Jira:**

```bash
python3 scripts/fetch_epic_issues.py
```

Queries each epic in `config.yaml` for its labeled issues and writes them to `scripts/epic_training_data.json`.

**Step 2 — Retrain:**

```bash
python3 scripts/retrain_model.py
```

Runs a grid search (72 configurations) with 5-fold stratified cross-validation, picks the best by macro F1, trains a final model on all data, and writes:

- `epic_classifier_model.pkl` — the trained pipeline, loaded automatically at runtime
- `navigation_epic_lookup.json` — top features per epic + keyword fallback rules

The grid explores:

| Parameter | Range |
|-----------|-------|
| SVM regularisation `C` | 0.5, 1.0, 2.0 |
| Word TF-IDF vocab | 15k, 25k tokens |
| Char TF-IDF vocab | 8k, 15k tokens |
| Title branch weight | 1.0–2.0 |
| Description branch weight | 1.0–1.5 |
| Char branch weight | 0.2, 0.4 |

A baseline (single-branch TF-IDF + LinearSVC) is also evaluated so you can see what the 3-branch architecture actually buys you.

### Adding a new epic

Add the key → name entry to `config.yaml` under `epics:`, then retrain. The model file and lookup table will pick it up automatically.

---

## Audit log

`epic_choices_log.csv` is created on first interactive use and appended on every decision:

| Column | Description |
|--------|-------------|
| Datetime | When the decision was made |
| Jira Key | The ticket |
| Suggested Epic | What the model predicted |
| Score | Raw decision function value (higher = more confident) |
| Accepted Epic | What was actually assigned |
| Accepted | `True` if the suggestion was taken, `False` if overridden |

The `Accepted` column is your model quality signal over time. If a particular epic is consistently overridden, it's a sign the class needs more training data or a clearer boundary from its neighbours.

---

## File reference

| File | Purpose |
|------|---------|
| `epic_matcher.py` | CLI entry point and importable API (`classify_ticket`, `fetch_and_classify`) |
| `config.default.yaml` | Template config — copy to `config.yaml` and fill in your values |
| `config.py` | Loads `config.yaml`, falls back to `config.default.yaml` |
| `jirahelper.py` | Jira REST API wrapper — connect, search, write Epic Link, manage comments |
| `authhelper.py` | Reads API tokens from `.netrc` / `_netrc` |
| `correct_logs.py` | Inspect the full Epic Link change history for a list of tickets |
| `navigation_epic_lookup.json` | Generated — keyword fallback rules and top model features per epic |
| `scripts/fetch_epic_issues.py` | Step 1: pull labeled training data from Jira |
| `scripts/retrain_model.py` | Step 2: grid search + retrain + save model |
| `scripts/model_components.py` | `FieldExtractor` — custom sklearn transformer for the 3-branch pipeline |

---

## Model performance

Trained on navigation defect tickets (English + German, ~25 engineering epics):

| Metric | Value |
|--------|-------|
| Algorithm | LinearSVC, balanced class weights |
| CV accuracy | ~74% (5-fold stratified) |
| Macro F1 | ~63% |
| Training samples | ~6,700 labeled issues |
| Classes | 25+ epics |
| Vocab (word) | 25,000 tokens, 1–2 grams, lemmatized |
| Vocab (char) | 8,000 tokens, 3–5 char grams |

Accuracy varies by epic. High-signal classes (speech, positioning, charging station search) score above 85%. Structurally overlapping classes (multiple "guiding" epics) are harder and will benefit the most from adding training data.
