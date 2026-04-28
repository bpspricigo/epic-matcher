#!/usr/bin/env python3
"""
Epic Link Classifier for Navigation Bug Tickets

Uses a trained LinearSVC model (epic_classifier_model.pkl) to predict
the most likely Epic Link for navigation tickets.
Falls back to keyword-based matching if sklearn is not available.

Training: TF-IDF (1,2)-gram + LinearSVC on 5962 tickets (all issues from 14 epics).
Features: Title (5x) + extracted Expected/Observed sections + cleaned description (first 800 chars).
Cross-validated accuracy: ~76% (5-fold stratified).

Usage:
    # Fetch from Jira and classify all unassigned navigation tickets:
    python3 epic_matcher.py --jira

    # Fetch from Jira with custom JQL:
    python3 epic_matcher.py --jira --jql 'project = NAVBUG AND ...'

    # Limit results:
    python3 epic_matcher.py --jira --max 100

    # Match a single ticket by title/description:
    python3 epic_matcher.py "Ticket title here" "Optional description here"

    # Or import and use programmatically:
        from epic_matcher import classify_ticket
        epic_key, epic_name, confidence = classify_ticket("CCP drift while driving")
"""
import json
import re
import os
import sys
import pickle
import csv
from datetime import datetime

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "epic_classifier_model.pkl")
LOOKUP_TABLE_PATH = os.path.join(_DIR, "navigation_epic_lookup.json")
LOG_FILE_PATH = os.path.join(_DIR, "epic_choices_log.csv")

from scripts.model_components import FieldExtractor
from config import jira as _jira_cfg, epics as EPIC_NAMES

DEFAULT_JQL = (
    f'filter = "{_jira_cfg["default_filter"]}" '
    'AND "Epic Link" is EMPTY'
)

# ── spaCy setup (optional) ─────────────────────────────────────────────────────
def _load_spacy():
    """Load EN + DE spaCy models if available. Returns (nlp_en, nlp_de) or (None, None)."""
    try:
        import spacy
        nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner"])
        print("spaCy loaded: EN + DE lemmatization enabled")
        return nlp_en, nlp_de
    except Exception as e:
        print(f"spaCy not available ({e}) — lemmatization disabled")
        return None, None

NLP_EN, NLP_DE = _load_spacy()

# Stopwords
_STOP_EN = {
    "the","a","an","is","it","in","of","to","and","or","not","with","for",
    "on","at","be","are","was","were","has","have","had","this","that","from",
    "by","as","but","no","if","so","do","did","can","will","would","should",
    "could","may","might","its","their","which","when","where","what","how",
}
_STOP_DE = {
    "der","die","das","ein","eine","ist","in","von","zu","und","oder","nicht",
    "mit","für","auf","bei","an","es","er","sie","wir","als","aus","nach",
    "wird","wurde","werden","hat","haben","hatte","war","waren","auch","noch",
    "sich","dem","den","des","im","am","zum","zur","dieser","diese","dieses",
}
STOPWORDS = _STOP_EN | _STOP_DE


def lemmatize(text):
    """
    Runs both spaCy EN and DE models on the same text and pools the results.
    """
    if NLP_EN is None and NLP_DE is None:
        return text

    tokens = []
    for nlp in [NLP_EN, NLP_DE]:
        if nlp is None:
            continue
        doc = nlp(text[:50_000])
        tokens.extend([
            t.lemma_.lower()
            for t in doc
            if not t.is_space
            and not t.is_punct
            and len(t.text) > 2
            and t.lemma_.lower() not in STOPWORDS
        ])
    return ' '.join(tokens)

# ── Feature flag ──
# Set to True to actually write the accepted Epic Link back to Jira.
# When False the tool is fully read-only (safe for testing / dry-runs).
CHANGE_JIRA_DATA = True

# ── Lazy-load the sklearn model ──
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                _model = pickle.load(f)
            return _model
        except Exception as e:
            print(f"Warning: Could not load model: {e}", file=sys.stderr)
    return None

# ── CSV Logging ──

def log_choice(jira_key, suggested_epic, score, accepted_epic, acceptedBol):
    """
    Log a choice to the CSV file.
    
    Args:
        jira_key: The Jira ticket key (e.g., "EPIC-123456")
        suggested_epic: The epic key suggested by the model (e.g., "EPIC-342350")
        score: The confidence score from the model
        accepted_epic: The epic key that was actually accepted/chosen
        acceptedBol: True or false for first suggestion acceptance
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(LOG_FILE_PATH)
    
    try:
        with open(LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Datetime', 'Jira Key', 'Suggested Epic', 'Score', 'Accepted Epic', 'Accepted']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write the log entry
            writer.writerow({
                'Datetime': timestamp,
                'Jira Key': jira_key,
                'Suggested Epic': suggested_epic,
                'Score': score,
                'Accepted Epic': accepted_epic,
                'Accepted': acceptedBol
            })
        
        print(f"  [Log] Choice logged to {LOG_FILE_PATH}")
    except Exception as exc:
        print(f"  [Log] ERROR writing to log file: {exc}", file=sys.stderr)

# ── Description extraction ──

_START_RE = re.compile(
    r'expected\s+result\s*/?\s*behavio(?:u)?r\s*:|'
    r'observed\s+result\s*/?\s*behavio(?:u)?r\s*:|'
    r'expected\s+result\s*/?\s*:\s*|'
    r'observed\s+result\s*/?\s*:\s*|'
    r'actual\s+result\s*/?\s*:\s*|'
    r'observed\s+behavio(?:u)?r\s*:|'
    r'expected\s+behavio(?:u)?r\s*:|'
    r'erwartetes?\s+(?:ergebnis|verhalten)\s*:|'
    r'(?:tatsächliches?|beobachtetes?)\s+(?:ergebnis|verhalten)\s*:|'
    r'issue\s+observed\s*:',
    re.IGNORECASE
)

_STOP_RE = re.compile(
    r'possible\s+impact|'
    r'actions?\s*/?\s*steps?\s+to\s+re(?:cover|produce)|'
    r'test\s+case\s+reference|'
    r'defect\s+severity|'
    r'occur\s+times|'
    r'links\s*:|'
    r'portal-access|'
    r'customer\s+(?:irritated|annoyed)|'
    r'platform\s+(?:responsible|e\.g)|'
    r'observed\s+behavio(?:u)?r\s+occurs|'
    r'preconditions?\s+(?:and)?\s*actions?|'
    r'steps\s+to\s+(?:reproduce|recovery)|'
    r'error\s+occurrence|'
    r'actual\s+driver|'
    r'\*\*timestamp|'
    r'\*\*reporter|'
    r'\*\*vehicle|'
    r'\*\*trip|'
    r'\*\*int-level|'
    r'\*\*head\s+unit|'
    r'defect-tracker|'
    r'equipment\s*:|'
    r'connected-services|'
    r'ecu\s*:|'
    r'map\s+db\s*:|'
    r'sys-approval|'
    r'\{code|'
    r'traceback\s+\(most\s+recent',
    re.IGNORECASE
)

_SKIP_LINES = frozenset([
    '{code}', '{noformat}', '{panel}', '----', '-', 'no', 'yes', 'none'
])


def extract_expected_observed(description):
    """
    Extract the 'Expected Result/Behaviour' and 'Observed Result/Behaviour'
    sections from a Jira ticket description, ignoring metadata noise.
    Returns the extracted text (max 500 chars) or empty string.
    """
    if not description:
        return ""
    text = description.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    parts = []
    capturing = False
    block = []

    for line in lines:
        s = line.strip()
        sl = s.lower()
        if _STOP_RE.search(sl):
            if capturing and block:
                parts.append(' '.join(block))
                block = []
            capturing = False
            continue
        if _START_RE.search(sl):
            if capturing and block:
                parts.append(' '.join(block))
                block = []
            capturing = True
            rem = _START_RE.sub('', s, count=1).strip()
            if rem and len(rem) > 3:
                block.append(rem)
            continue
        if capturing and s and len(s) > 2:
            if s.startswith('http') or s.startswith('\\\\'):
                continue
            if s in _SKIP_LINES:
                continue
            if s.startswith('|') or s.startswith('||'):
                continue
            block.append(s)

    if capturing and block:
        parts.append(' '.join(block))
    return ' '.join(parts)[:500]


def clean_description(description, max_chars=800):
    """Clean first N chars of full description, removing Jira markup and URLs."""
    if not description:
        return ""
    raw = description[:max_chars].lower()
    raw = re.sub(r'\{[^}]+\}', '', raw)   # remove Jira markup {code}, {noformat} etc.
    raw = re.sub(r'http\S+', '', raw)      # remove URLs
    raw = re.sub(r'\*+', '', raw)          # remove bold/italic markers
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw


def prepare_text(title, description=""):
    """Build model input: title (5x) + extracted expected/observed + cleaned description."""
    extracted = extract_expected_observed(description)
    parts = [title] * 5
    if extracted:
        parts.append(extracted)
    parts.append(clean_description(description, 800))
    return ' '.join(parts)


# ── Classification ──

def classify_ticket(title, description=""):
    """
    Classify a single ticket using the sklearn model.
    Returns (epic_key, epic_name, confidence_score, alternatives).
    Falls back to keyword matching if sklearn model is unavailable.
    """
    model = _load_model()
    if model is not None:
        # Preprocess into the format the model expects
        # The model was trained on dicts with preprocessed fields
        preprocessed_data = preprocess_for_model(title, description)
        
        # decision_function gives per-class scores
        scores = model.decision_function([preprocessed_data])[0]
        best_idx = scores.argmax()
        epic_key = model.classes_[best_idx]
        epic_name = EPIC_NAMES.get(epic_key, epic_key)
        confidence = float(scores[best_idx])

        # Build top-3 alternatives
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        alternatives = []
        for idx, sc in ranked[:3]:
            ek = model.classes_[idx]
            alternatives.append({
                "epic_key": ek,
                "epic_name": EPIC_NAMES.get(ek, ek),
                "score": round(float(sc), 2),
            })

        return epic_key, epic_name, round(confidence, 2), alternatives
    else:
        # Fallback: keyword-based matching
        results = _match_epic_keywords(title, description)
        if results:
            best = results[0]
            return best["epic_key"], best["epic_name"], best["score"], results[:3]
        return "-", "NO MATCH", 0, []


def preprocess_for_model(summary, description=""):
    """
    Preprocess a single ticket into the format expected by the trained model.
    The model expects dicts with these preprocessed fields:
      - _summary_clean: cleaned + lemmatized summary
      - _desc_eo_lemmatized: EO text, cleaned + lemmatized
      - _desc_eo_clean: EO text, cleaned but NOT lemmatized
    """
    # Extract EO sections
    eo_raw = extract_expected_observed(description)
    eo_text = eo_raw if eo_raw.strip() else description
    
    # Apply cleaning and lemmatization
    # Note: You need to have lemmatize() function available
    # If spaCy is not available, lemmatize() will just return the text as-is
    summary_clean = lemmatize(clean_description(summary, max_chars=300))
    desc_eo_lemmatized = lemmatize(clean_description(eo_text, max_chars=1000))
    desc_eo_clean = clean_description(eo_text, max_chars=600)
    
    return {
        'summary': summary,
        'description': description,
        '_summary_clean': summary_clean,
        '_desc_eo_lemmatized': desc_eo_lemmatized,
        '_desc_eo_clean': desc_eo_clean,
    }


def _match_epic_keywords(title, description=""):
    """Fallback keyword-based matcher using the JSON lookup table."""
    try:
        with open(LOOKUP_TABLE_PATH, 'r', encoding='utf-8') as f:
            lookup = json.load(f)
    except FileNotFoundError:
        return []

    title_lower = title.lower()
    desc_lower = (description or "").lower()
    combined = title_lower + " " + desc_lower
    scores = {}

    for rule in lookup.get("keyword_rules", []):
        epic_key = rule["epic_key"]
        epic_name = rule["epic_name"]
        score = 0

        for pattern in rule.get("title_patterns", []):
            pl = pattern.lower()
            if pl in title_lower:
                score += 10
            elif pl in desc_lower:
                score += 3

        for pattern in rule.get("description_patterns", []):
            if pattern.lower() in combined:
                score += 2

        if score > 0:
            scores[epic_key] = {
                "epic_key": epic_key,
                "epic_name": epic_name,
                "score": score,
            }

    return sorted(scores.values(), key=lambda x: x["score"], reverse=True)


# ── Jira integration ──

def fetch_and_classify(jql=None, max_results=200):
    """
    Fetch navigation tickets without Epic Link from Jira and classify them.
    Returns list of dicts: ticket_key, summary, epic_key, epic_name, score, alternatives.
    """
    import jirahelper

    jirahelper.set_use_primary_jira(True)
    jira = jirahelper.get_jira()

    query = jql or DEFAULT_JQL
    model = _load_model()
    mode = "sklearn" if model is not None else "keyword-fallback"
    print(f"Classifier: {mode}")
    print(f"JQL: {query}")
    print(f"Fetching tickets from Jira...")

    classified = []
    start_at = 0
    batch_size = 50

    while start_at < max_results:
        issues = jira.search_issues(query, startAt=start_at, maxResults=batch_size,
                                    fields='summary,description')
        if not issues:
            break

        for issue in issues:
            key = issue.key
            summary = issue.fields.summary or ""
            description = issue.fields.description or ""

            # classify_ticket now handles preprocessing internally
            epic_key, epic_name, score, alts = classify_ticket(summary, description)
            
            classified.append({
                "ticket_key": key,
                "summary": summary,
                "description": description,
                "epic_key": epic_key,
                "epic_name": epic_name,
                "score": score,
                "alternatives": alts,
            })

        start_at += len(issues)
        print(f"  fetched {start_at} tickets...")

        if len(issues) < batch_size:
            break

    return classified


# ── Output ──

def print_table(classified):
    """Print a formatted table of classified tickets."""
    print()
    print(f"{'#':<4} {'Ticket':<22} {'Title':<80} {'| Epics in order':<35}")
    print("-" * 260)

    classified_sorted = sorted(classified, key=lambda r: r["epic_name"] or "")

    for i, row in enumerate(classified_sorted, 1):
        title_short = row["summary"][:70]
        alts = row["alternatives"]
        alts_sorted = sorted(alts, key=lambda x: x["score"], reverse=True)
        # Format: rank, epic name, score with sign, fixed width
        line = " | ".join(
            f"{rank+1:<2} {alt['epic_name']:<35} ({alt['score']:+5.2f})"
            for rank, alt in enumerate(alts_sorted)
        )
        print(f"{i:<4} {row['ticket_key']:<22} {title_short:<80} {'|'}{line} ")


    total = len(classified)
    matched = sum(1 for r in classified if r["epic_key"] != "-")
    no_match = total - matched
    print("-" * 260)
    print(f"Total: {total}  |  Classified: {matched}  |  No match: {no_match}")

    # Epic distribution
    epic_counts = {}
    for r in classified:
        name = r["epic_name"]
        epic_counts[name] = epic_counts.get(name, 0) + 1
    print(f"\nEpic distribution:")
    for name, count in sorted(epic_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<35} {count}")


def print_grouped_table(classified, target_epic):
    """
    Print tickets grouped by predicted epic, sorted by score within each group.
    The group whose name contains `target_epic` (case-insensitive) is printed first.
    """
    # Build groups: epic_name -> [rows sorted by score desc]
    groups = {}
    for row in classified:
        name = row["epic_name"]
        groups.setdefault(name, []).append(row)
    for name in groups:
        groups[name].sort(key=lambda r: r["score"], reverse=True)

    # Find matching group name (case-insensitive substring)
    target_lower = target_epic.lower()
    matched_name = next(
        (n for n in groups if target_lower in n.lower()),
        None,
    )
    if matched_name is None:
        print(f"Warning: no tickets were classified under an epic matching '{target_epic}'.")
        print(f"Known epic names: {', '.join(sorted(groups.keys()))}\n")

    # Order: matching group first, then the rest alphabetically
    other_names = sorted(n for n in groups if n != matched_name)
    ordered_names = ([matched_name] if matched_name else []) + other_names

    ticket_num = 0
    total = len(classified)
    header = f"{'#':<4} {'Ticket':<22} {'Score':<8} {'Title':<72} {'| Epics in order'}"
    separator = "-" * 260

    for group_idx, name in enumerate(ordered_names):
        rows = groups[name]
        label = f" GROUP: {name} ({len(rows)} ticket(s)) "
        if name == matched_name:
            label += "  << requested epic"
        print()
        print("#" * 72)
        print(f"#{label}")
        print("#" * 72)
        print(header)
        print(separator)

        for row in rows:
            ticket_num += 1
            title_short = row["summary"][:62]
            alts = sorted(row["alternatives"], key=lambda x: x["score"], reverse=True)
            alts_line = " | ".join(
                f"{r+1:<2} {alt['epic_name']:<35} ({alt['score']:+5.2f})"
                for r, alt in enumerate(alts)
            )
            print(f"{ticket_num:<4} {row['ticket_key']:<22} {row['score']:+6.2f}  {title_short:<72} |{alts_line}")

        print(separator)

    matched = sum(1 for r in classified if r["epic_key"] != "-")
    print(f"\nTotal: {total}  |  Classified: {matched}  |  No match: {total - matched}")
    if matched_name:
        print(f"Showing group '{matched_name}' first ({len(groups[matched_name])} tickets).")


# ── Jira write helper ──

_epic_link_field_id = None  # cached after first lookup


def _resolve_epic_link_field(jira):
    """
    Discover the correct custom field ID for 'Epic Link' on this Jira instance.
    Different server / data-center installations use different IDs.
    The result is cached in _epic_link_field_id for the session.
    """
    global _epic_link_field_id
    if _epic_link_field_id is not None:
        return _epic_link_field_id

    try:
        all_fields = jira.fields()
    except Exception as exc:
        print(f"  [Jira] Could not fetch field list: {exc}", file=sys.stderr)
        return None

    # Look for a field whose name is exactly (or closely) 'Epic Link'
    candidates = [f for f in all_fields if "epic link" in f["name"].lower()]
    if not candidates:
        names = [f["name"] for f in all_fields if "epic" in f["name"].lower()]
        print(
            f"  [Jira] WARNING: No 'Epic Link' field found. "
            f"Epic-related fields on this instance: {names}",
            file=sys.stderr,
        )
        return None

    # Prefer exact match, otherwise take first partial match
    exact = [f for f in candidates if f["name"].lower() == "epic link"]
    field = exact[0] if exact else candidates[0]
    _epic_link_field_id = field["id"]
    print(f"  [Jira] Resolved Epic Link field: '{field['name']}' → {_epic_link_field_id}")
    return _epic_link_field_id


def set_epic_link(jira, ticket_key, epic_key):
    """
    Write the Epic Link field on a Jira ticket.
    The field ID is discovered at runtime so the code works across different
    Jira server / data-center instances.
    Only called when CHANGE_JIRA_DATA is True.
    """
    field_id = _resolve_epic_link_field(jira)
    if field_id is None:
        print(f"  [Jira] Skipping update for {ticket_key} – Epic Link field unknown.", file=sys.stderr)
        return
    try:
        issue = jira.issue(ticket_key)
        issue.update(fields={field_id: epic_key})
        print(f"  [Jira] Epic Link set on {ticket_key} → {epic_key}")
    except Exception as exc:
        print(f"  [Jira] ERROR updating {ticket_key}: {exc}", file=sys.stderr)


# ── Interactive mode ──

def interactive_mode(classified):
    """
    Interactively review each ticket: confirm the suggested epic or pick a new one.
    Prints a counter (X of Y) for each ticket.
    """
    try:
        import questionary
    except ImportError:
        print(
            "Error: 'questionary' is required for --interactively.\n"
            "Install it with: pip install questionary",
            file=sys.stderr,
        )
        sys.exit(1)

    total = len(classified)
    epic_choices = [
        questionary.Choice(
            title=f"{name:<40}  {key}",
            value=key,
        )
        for key, name in sorted(EPIC_NAMES.items(), key=lambda x: x[1])
    ]

    # Lazily obtain the Jira client once if we are going to write data
    _jira = None
    if CHANGE_JIRA_DATA:
        try:
            import jirahelper
            jirahelper.set_use_primary_jira(True)
            _jira = jirahelper.get_jira()
        except Exception as exc:
            print(f"Warning: could not connect to Jira for writing: {exc}", file=sys.stderr)
            print("Continuing in read-only mode.", file=sys.stderr)

    for idx, row in enumerate(classified, 1):
        remaining = total - idx + 1
        print()
        print("=" * 72)
        print(f"  Ticket {idx} of {total}  |  {remaining} ticket(s) left (including this one)")
        print("=" * 72)
        print(f"  Key   : {row['ticket_key']}")
        print(f"  Title : {row['summary']}")

        # Show relevant description extract, falling back to cleaned description
        snippet = extract_expected_observed(row.get("description", ""))
        if not snippet:
            snippet = clean_description(row.get("description", ""), 400)
        if snippet:
            # Wrap long snippet at ~70 chars for readability
            wrapped = "\n    ".join(
                snippet[i : i + 70] for i in range(0, min(len(snippet), 420), 70)
            )
            print(f"\n  Description:\n    {wrapped}")

        print()
        print(f"  Suggested Epic : {row['epic_name']}  ({row['epic_key']})")
        print(f"  Score          : {row['score']:+.2f}")
        print()

        action = questionary.select(
            "  What do you want to do?",
            choices=[
                questionary.Choice(title="Accept suggested epic", value="accept"),
                questionary.Choice(title="Choose a different epic", value="choose"),
                questionary.Choice(title="Skip this ticket", value="skip"),
            ],
            default="accept",
        ).ask()

        if action is None:          # user pressed Ctrl-C / EOF
            print("\nAborted.")
            sys.exit(0)

        if action == "accept":
            print(f"\n  ACCEPTED: {row['epic_key']} - {row['epic_name']}")
            if CHANGE_JIRA_DATA and _jira:
                set_epic_link(_jira, row['ticket_key'], row['epic_key'])
                # Log the choice
                log_choice(
                    jira_key=row['ticket_key'],
                    suggested_epic=row['epic_key'],
                    score=row['score'],
                    accepted_epic=row['epic_key'],
                    acceptedBol=True
                )
        elif action == "choose":
            chosen_key = questionary.select(
                "  Select epic to assign:",
                choices=epic_choices,
            ).ask()

            if chosen_key is None:    # user pressed Ctrl-C / EOF
                print("\nAborted.")
                sys.exit(0)

            chosen_name = EPIC_NAMES.get(chosen_key, chosen_key)
            print(f"\n  ACCEPTED: {chosen_key} - {chosen_name}")
            if CHANGE_JIRA_DATA and _jira:
                set_epic_link(_jira, row['ticket_key'], chosen_key)
                # Log the choice
                log_choice(
                    jira_key=row['ticket_key'],
                    suggested_epic=row['epic_key'],
                    score=row['score'],
                    accepted_epic=chosen_key,
                    acceptedBol=False
                )
        else:  # skip
            print("\n  SKIPPED")

    print()
    print("=" * 72)
    print(f"  Done – reviewed {total} ticket(s).")
    print("=" * 72)


def main():
    if "--jira" in sys.argv:
        jql = None
        if "--jql" in sys.argv:
            jql_idx = sys.argv.index("--jql")
            if jql_idx + 1 < len(sys.argv):
                jql = sys.argv[jql_idx + 1]

        max_results = 200
        if "--max" in sys.argv:
            max_idx = sys.argv.index("--max")
            if max_idx + 1 < len(sys.argv):
                max_results = int(sys.argv[max_idx + 1])

        classified = fetch_and_classify(jql=jql, max_results=max_results)

        target_epic = None
        if "--epic" in sys.argv:
            epic_idx = sys.argv.index("--epic")
            next_val = (
                sys.argv[epic_idx + 1]
                if epic_idx + 1 < len(sys.argv) and not sys.argv[epic_idx + 1].startswith("--")
                else None
            )
            if next_val:
                target_epic = next_val
            else:
                # No value provided – ask interactively
                try:
                    import questionary
                except ImportError:
                    print(
                        "Error: 'questionary' is required for the epic selection prompt.\n"
                        "Install it with: pip install questionary",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                epic_choices = [
                    questionary.Choice(title=f"{name}  ({key})", value=name)
                    for key, name in sorted(EPIC_NAMES.items(), key=lambda x: x[1])
                ]
                target_epic = questionary.select(
                    "Select the target epic to focus on:",
                    choices=epic_choices,
                ).ask()
                if target_epic is None:   # Ctrl-C
                    print("\nAborted.")
                    sys.exit(0)

        if "--interactively" in sys.argv:
            if target_epic:
                # Filter / order for interactive mode: target group first
                target_lower = target_epic.lower()
                in_group = [r for r in classified if target_lower in r["epic_name"].lower()]
                others   = [r for r in classified if target_lower not in r["epic_name"].lower()]
                in_group.sort(key=lambda r: r["score"], reverse=True)
                others.sort(key=lambda r: r["score"], reverse=True)
                classified = in_group + others
            interactive_mode(classified)
        elif target_epic:
            print_grouped_table(classified, target_epic)
        else:
            print_table(classified)
        return

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 epic_matcher.py --jira                          Fetch & classify from Jira")
        print("  python3 epic_matcher.py --jira --jql 'JQL...'           Custom JQL query")
        print("  python3 epic_matcher.py --jira --max 100                Limit results")
        print("  python3 epic_matcher.py --jira --interactively          Review tickets interactively")
        print("  python3 epic_matcher.py --jira --epic 'Bug - Map'        Group by epic, target first (or omit value for selection list)")
        print('  python3 epic_matcher.py "title" ["description"]         Match single ticket')
        sys.exit(1)

    title = sys.argv[1]
    description = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"Title: {title}")
    if description:
        print(f"Description: {description[:100]}...")
    print()

    epic_key, epic_name, score, alternatives = classify_ticket(title, description)

    print("=== Epic Link Prediction ===")
    print(f"{'Rank':<5} {'Score':<8} {'Epic Key':<16} {'Epic Name'}")
    print("-" * 60)
    for i, alt in enumerate(alternatives, 1):
        marker = " <<<" if i == 1 else ""
        print(f"{i:<5} {alt['score']:<8} {alt['epic_key']:<16} {alt['epic_name']}{marker}")

    print(f"\n>>> Predicted: {epic_key} ({epic_name})  score={score}")


if __name__ == "__main__":
    main()
