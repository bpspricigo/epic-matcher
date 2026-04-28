#!/usr/bin/env python3
"""
retrain_model_v3.py — Epic Classifier (v3)

Changes vs v2:
  1. EO extraction done once upstream, stored in data dicts
  2. Lemmatization runs both NLP_EN + NLP_DE, pools tokens from both
  3. Explicit slice before spaCy call (defensive)
  4. Redundant cross_val_predict removed — grid search yp reused
  5. Baseline uses same cleaning functions as v3 (trustworthy comparison)
  6. Char branch weight added to grid search
  7. Char branch receives cleaned but un-lemmatized text

Usage:
    python3 scripts/retrain_model_v3.py

Input:
    scripts/epic_training_data.json

Output:
    epic_classifier_model.pkl
    navigation_epic_lookup.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json, re, pickle, time
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, f1_score

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
SCRIPT_DIR  = os.path.dirname(__file__)

from scripts.model_components import FieldExtractor
from config import epics as EPIC_NAMES

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

# English + German combined stopwords
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

# ── Jira markup cleaner ────────────────────────────────────────────────────────
_JIRA_RULES = [
    # Block macros with content: {code}...{code}, {noformat}...{noformat}
    (re.compile(r'\{(?:code|noformat)[^}]*\}.*?\{(?:code|noformat)\}', re.S | re.I), ' '),
    # Panel/color/other block macros with params: {panel:title=Foo}
    (re.compile(r'\{[a-z]+:[^}]+\}', re.I), ' '),
    # Simple macros: {code}, {noformat}, {panel}, {quote}, etc.
    (re.compile(r'\{[a-z]+\}', re.I), ' '),
    # Table rows: ||header||header|| or |cell|cell|
    (re.compile(r'^\s*\|+.*\|+\s*$', re.M), ' '),
    # Named links: [Link text|http://...]
    (re.compile(r'\[([^\]|]+)\|[^\]]+\]'), r'\1'),
    # Plain links: [http://...] or bare URLs
    (re.compile(r'\[?https?://\S+\]?'), ' '),
    # Image attachments: !image.png! or !image.png|thumbnail!
    (re.compile(r'!\S+!'), ' '),
    # @mentions
    (re.compile(r'@\w+'), ' '),
    # Bold/italic/underline: *text*, _text_, +text+, -text-, ^text^, ~text~
    (re.compile(r'[*_+\-^~](\S[^*_+\-^~]*\S)[*_+\-^~]'), r'\1'),
    # Heading markers: h1. h2. etc.
    (re.compile(r'^h[1-6]\.\s*', re.M), ' '),
    # List markers: * item, # item, ** item
    (re.compile(r'^[*#]+\s+', re.M), ' '),
    # Horizontal rules
    (re.compile(r'^----+\s*$', re.M), ' '),
]

def strip_jira_markup(text: str) -> str:
    for pattern, replacement in _JIRA_RULES:
        text = pattern.sub(replacement, text)
    return text

# ── Domain noise normalization ─────────────────────────────────────────────────
_NORM_RULES = [
    # Version strings: 23.03.1, 2.4.11, v1.2.3
    (re.compile(r'\bv?\d+\.\d+[\.\d]*\b'), 'VERSION'),
    # Hex error codes: 0x1A2B, 0xDEAD
    (re.compile(r'\b0x[0-9a-fA-F]+\b'), 'ERRCODE'),
    # Pure number tokens (coordinates, IDs, counts)
    (re.compile(r'\b\d{3,}\b'), 'NUM'),
    # GPS-style coordinates
    (re.compile(r'\b\d+\.\d{4,}\b'), 'COORD'),
    # Integration-level references
    (re.compile(r'int[-\s]?level\s*[\d.]*', re.I), 'INTLEVEL'),
    # ECU/HU references
    (re.compile(r'\b(?:ecu|head.?unit|hu)\s*[\d.]*\b', re.I), 'ECU'),
]

def normalize_domain(text: str) -> str:
    for pattern, replacement in _NORM_RULES:
        text = pattern.sub(replacement, text)
    return text

# ── Core text cleaning (no lemmatization — used by both branches) ──────────────
def clean_text(text: str, max_chars: int = 1000) -> str:
    if not text:
        return ""
    text = strip_jira_markup(text)
    text = normalize_domain(text)
    text = text[:max_chars].lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Lemmatization: runs EN + DE, pools tokens from both ───────────────────────
def lemmatize(text: str) -> str:
    """
    Runs both spaCy EN and DE models on the same text and pools the results.
    Each model lemmatizes what it recognizes; garbage tokens for the wrong
    language are filtered out by the stopword check and TF-IDF min_df.
    Ideal for mixed DE/EN defect reports where EO sections are often
    bilingual translations of each other.
    """
    if NLP_EN is None and NLP_DE is None:
        return text

    tokens = []
    for nlp in [NLP_EN, NLP_DE]:
        if nlp is None:
            continue
        # Fix 3: explicit slice before spaCy call (defensive against edge cases)
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

# ── EO section extraction (your original logic, kept intact) ──────────────────
_START_RE = re.compile(
    r'expected\s+result\s*/?\s*behavio(?:u)?r\s*:|'
    r'observed\s+result\s*/?\s*behavio(?:u)?r\s*:|'
    r'expected\s+result\s*/?\s*:\s*|observed\s+result\s*/?\s*:\s*|'
    r'actual\s+result\s*/?\s*:\s*|observed\s+behavio(?:u)?r\s*:|'
    r'expected\s+behavio(?:u)?r\s*:|'
    r'erwartetes?\s+(?:ergebnis|verhalten)\s*:|'
    r'(?:tatsächliches?|beobachtetes?)\s+(?:ergebnis|verhalten)\s*:|'
    r'issue\s+observed\s*:', re.IGNORECASE)

_STOP_RE = re.compile(
    r'possible\s+impact|actions?\s*/?\s*steps?\s+to\s+re(?:cover|produce)|'
    r'test\s+case\s+reference|defect\s+severity|occur\s+times|links\s*:|'
    r'portal-access|customer\s+(?:irritated|annoyed)|'
    r'platform\s+(?:responsible|e\.g)|'
    r'observed\s+behavio(?:u)?r\s+occurs|preconditions?\s+(?:and)?\s*actions?|'
    r'steps\s+to\s+(?:reproduce|recovery)|error\s+occurrence|actual\s+driver|'
    r'\*\*timestamp|\*\*reporter|\*\*vehicle|\*\*trip|\*\*int-level|\*\*head\s+unit|'
    r'defect-tracker|equipment\s*:|connected-services|ecu\s*:|'
    r'map\s+db\s*:|sys-approval|\{code|traceback\s+\(most\s+recent', re.IGNORECASE)

_SKIP_LINES = frozenset(['{code}', '{noformat}', '{panel}', '----', '-', 'no', 'yes', 'none'])

def extract_eo(desc: str) -> str:
    if not desc:
        return ""
    text = desc.replace('\r\n', '\n').replace('\r', '\n')
    parts, cap, blk = [], False, []
    for line in text.split('\n'):
        s  = line.strip()
        sl = s.lower()
        if _STOP_RE.search(sl):
            if cap and blk:
                parts.append(' '.join(blk))
                blk = []
            cap = False
            continue
        if _START_RE.search(sl):
            if cap and blk:
                parts.append(' '.join(blk))
                blk = []
            cap = True
            rem = _START_RE.sub('', s, count=1).strip()
            if rem and len(rem) > 3:
                blk.append(rem)
            continue
        if cap and s and len(s) > 2:
            if s.startswith('http') or s.startswith('\\\\'):
                continue
            if s in _SKIP_LINES or s.startswith('|'):
                continue
            blk.append(s)
    if cap and blk:
        parts.append(' '.join(blk))
    return ' '.join(parts)[:600]

# ── Fix 1: pre-process data once upstream ─────────────────────────────────────
def preprocess_data(raw_data: list) -> list:
    """
    Runs EO extraction and cleaning once for all issues before any pipeline
    sees the data. Each dict gets three precomputed fields:
      - _summary_clean:     cleaned + lemmatized summary
      - _desc_eo_lemmatized: EO text, cleaned + lemmatized (word branch)
      - _desc_eo_clean:     EO text, cleaned but NOT lemmatized (char branch)

    This avoids redundant extraction across branches and CV folds.
    """
    processed = []
    for d in raw_data:
        summary = d.get('summary', '') or ''
        desc    = d.get('description', '') or ''

        eo_raw = extract_eo(desc)
        # Fall back to full description if EO extraction found nothing
        eo_text = eo_raw if eo_raw.strip() else desc

        summary_clean        = lemmatize(clean_text(summary, max_chars=300))
        desc_eo_lemmatized   = lemmatize(clean_text(eo_text, max_chars=1000))
        desc_eo_clean        = clean_text(eo_text, max_chars=600)   # un-lemmatized for char branch

        processed.append({
            **d,
            '_summary_clean':       summary_clean,
            '_desc_eo_lemmatized':  desc_eo_lemmatized,
            '_desc_eo_clean':       desc_eo_clean,
        })
    return processed

# ── Pipeline builder ───────────────────────────────────────────────────────────
def build_pipeline(C: float,
                   max_features_word: int,
                   max_features_char: int,
                   summary_weight: float,
                   desc_weight: float,
                   char_weight: float) -> Pipeline:
    """
    Three-branch FeatureUnion over precomputed fields:
      1. _summary_clean       → word TF-IDF  (lemmatized)
      2. _desc_eo_lemmatized  → word TF-IDF  (lemmatized)
      3. _desc_eo_clean       → char TF-IDF  (un-lemmatized, typo-robust)
    """
    union = FeatureUnion(
        transformer_list=[
            ('summary_word', Pipeline([
                ('extract', FieldExtractor('_summary_clean')),
                ('tfidf', TfidfVectorizer(
                    analyzer='word', ngram_range=(1, 2),
                    sublinear_tf=True, min_df=2,
                    max_features=max_features_word,
                )),
            ])),
            ('desc_word', Pipeline([
                ('extract', FieldExtractor('_desc_eo_lemmatized')),
                ('tfidf', TfidfVectorizer(
                    analyzer='word', ngram_range=(1, 2),
                    sublinear_tf=True, min_df=2,
                    max_features=max_features_word,
                )),
            ])),
            ('desc_char', Pipeline([
                ('extract', FieldExtractor('_desc_eo_clean')),   # Fix 8: un-lemmatized
                ('tfidf', TfidfVectorizer(
                    analyzer='char_wb', ngram_range=(3, 5),
                    sublinear_tf=True, min_df=3,
                    max_features=max_features_char,
                )),
            ])),
        ],
        transformer_weights={
            'summary_word': summary_weight,
            'desc_word':    desc_weight,
            'desc_char':    char_weight,    # Fix 7: now part of grid search
        }
    )
    return Pipeline([
        ('features', union),
        ('clf', LinearSVC(C=C, max_iter=10000, class_weight='balanced')),
    ])

# ── Fix 5+6: Baseline uses same cleaning, no redundant CV call ────────────────
def build_baseline_pipeline() -> Pipeline:
    """
    Reproduces the spirit of v1 — single concatenated field, single TF-IDF —
    but uses the same cleaning functions as v3 so the comparison is trustworthy.
    The only structural difference is no FeatureUnion and no char branch.
    """
    class BaselineConcatExtractor(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X):
            # Reuses precomputed fields — same cleaning as v3 word branches
            return [
                (d.get('_summary_clean', '') + ' ') * 3
                + (d.get('_desc_eo_lemmatized', '') or '')
                for d in X
            ]

    return Pipeline([
        ('extract', BaselineConcatExtractor()),
        ('tfidf',  TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2),
            sublinear_tf=True, min_df=2, max_features=20000,
        )),
        ('clf', LinearSVC(C=1.0, max_iter=10000, class_weight='balanced')),
    ])

# ── Load + preprocess data ─────────────────────────────────────────────────────
DATA_PATH = os.path.join(SCRIPT_DIR, 'epic_training_data.json')
print(f"Loading: {DATA_PATH}")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

print(f"Preprocessing {len(raw_data)} issues (EO extraction + cleaning + lemmatization)...")
t_pre = time.time()
data  = preprocess_data(raw_data)       # Fix 1: done once, reused everywhere
print(f"Preprocessing done in {time.time() - t_pre:.1f}s")

labels       = [d['epic_key'] for d in data]
label_counts = Counter(labels)

print(f"\nTotal samples: {len(labels)}")
print(f"\nClass distribution:")
for ek in sorted(label_counts, key=lambda x: -label_counts[x]):
    print(f"  {EPIC_NAMES.get(ek, ek):<35} {label_counts[ek]:>5}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Baseline ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"BASELINE")
print(f"{'='*80}")
t0      = time.time()
yp_base = cross_val_predict(build_baseline_pipeline(), data, labels, cv=cv)
t_base  = time.time() - t0
acc_base = 100 * sum(a == p for a, p in zip(labels, yp_base)) / len(labels)
f1_base  = 100 * f1_score(labels, yp_base, average='macro')
print(f"  Accuracy: {acc_base:.1f}%   Macro F1: {f1_base:.1f}%   Time: {t_base:.1f}s")

# ── Grid search ────────────────────────────────────────────────────────────────
configs = []
for C in [0.5, 1.0, 2.0]:
    for max_feat_word in [15000, 25000]:
        for max_feat_char in [8000, 15000]:
            for sw, dw in [(2.0, 1.0), (1.5, 1.0), (1.0, 1.5)]:
                for cw in [0.2, 0.4]:               # Fix 7: char weight in grid
                    configs.append((C, max_feat_word, max_feat_char, sw, dw, cw))

print(f"\n{'='*80}")
print(f"GRID SEARCH — {len(configs)} configs")
print(f"{'='*80}")
print(f"{'Config':<65} {'Acc':>6} {'F1':>6} {'Time':>6}")
print(f"{'-'*80}")

results     = []
best_yp     = None   # Fix 5: store yp from best config, avoid redundant CV call
best_so_far = -1.0
total_configs = len(configs)

for idx, (C, mfw, mfc, sw, dw, cw) in enumerate(configs, 1):
    pipe = build_pipeline(C, mfw, mfc, sw, dw, cw)
    t0   = time.time()
    yp   = cross_val_predict(pipe, data, labels, cv=cv)
    t    = time.time() - t0
    acc  = 100 * sum(a == p for a, p in zip(labels, yp)) / len(labels)
    f1   = 100 * f1_score(labels, yp, average='macro')

    name      = f"C={C}, word={mfw}, char={mfc}, sw={sw}, dw={dw}, cw={cw}"
    delta_acc = acc - acc_base
    delta_f1  = f1  - f1_base

    # Progress indicator
    progress = f"[{idx}/{total_configs}]"
    print(f"{progress:<10} {name:<65} {acc:>5.1f}% {f1:>5.1f}%  {t:>5.1f}s  "
          f"[Δacc {delta_acc:+.1f}  ΔF1 {delta_f1:+.1f}]")

    results.append((name, acc, f1, C, mfw, mfc, sw, dw, cw, yp))

    if f1 > best_so_far:
        best_so_far = f1
        best_yp     = yp

# Sort by macro F1
results.sort(key=lambda x: -x[2])

print(f"\n{'='*80}")
print(f"TOP 5  (sorted by Macro F1)")
print(f"{'='*80}")
print(f"  {'Baseline':<62} Acc={acc_base:.1f}%  F1={f1_base:.1f}%")
print(f"  {'-'*75}")
for name, acc, f1, *_ in results[:5]:
    delta_acc = acc - acc_base
    delta_f1  = f1  - f1_base
    print(f"  {name:<62} Acc={acc:.1f}%  F1={f1:.1f}%  "
          f"[Δacc {delta_acc:+.1f}  ΔF1 {delta_f1:+.1f}]")

# ── Train final model ──────────────────────────────────────────────────────────
best_name, best_acc, best_f1, C, mfw, mfc, sw, dw, cw, best_yp = results[0]
print(f"\n>>> Training final model: {best_name}")
print(f"    CV Accuracy: {best_acc:.1f}%   Macro F1: {best_f1:.1f}%")

target_names = [EPIC_NAMES[ek] for ek in sorted(set(labels))]
print(f"\nDetailed CV report (reusing grid search predictions — no extra CV call):")
print(classification_report(
    labels, best_yp,
    labels=sorted(set(labels)),
    target_names=target_names,
    digits=2,
))

final = build_pipeline(C, mfw, mfc, sw, dw, cw)
final.fit(data, labels)

# ── Save model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(PROJECT_DIR, 'epic_classifier_model.pkl')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(final, f)
print(f"Model saved: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024:.0f} KB)")

# ── Save lookup JSON ───────────────────────────────────────────────────────────
fu           = final.named_steps['features']
clf          = final.named_steps['clf']
all_features = []
for branch_name, pipe in fu.transformer_list:
    tfidf = pipe.named_steps['tfidf']
    all_features.extend([f"{branch_name}::{f}" for f in tfidf.get_feature_names_out()])

lookup = {
    "_meta": {
        "description":       "Navigation Epic Link lookup table",
        "source":            f"Trained on {len(labels)} issues — FeatureUnion TF-IDF + LinearSVC",
        "cv_accuracy":       best_acc,
        "cv_macro_f1":       best_f1,
        "baseline_accuracy": acc_base,
        "baseline_macro_f1": f1_base,
        "delta_accuracy":    round(best_acc - acc_base, 2),
        "delta_macro_f1":    round(best_f1  - f1_base,  2),
        "training_samples":  len(labels),
        "model_file":        "epic_classifier_model.pkl",
        "best_config":       best_name,
        "spacy_enabled":     NLP_EN is not None,
    },
    "epics":         {},
    "keyword_rules": [],
}

for i, ek in enumerate(clf.classes_):
    coefs    = clf.coef_[i]
    top_idx  = coefs.argsort()[-25:][::-1]
    top_feat = [(all_features[j], round(float(coefs[j]), 2)) for j in top_idx]

    lookup["epics"][ek] = {
        "epic_name":    EPIC_NAMES[ek],
        "ticket_count": label_counts[ek],
        "top_features": [f for f, _ in top_feat[:15]],
    }
    lookup["keyword_rules"].append({
        "epic_key":             ek,
        "epic_name":            EPIC_NAMES[ek],
        "priority":             i + 1,
        "title_patterns":       [f.split('::')[1] for f, _ in top_feat
                                 if 'summary' in f
                                 and ' ' not in f.split('::')[1]][:15],
        "description_patterns": [f.split('::')[1] for f, _ in top_feat
                                 if 'desc_word' in f][:10],
    })

LOOKUP_PATH = os.path.join(PROJECT_DIR, 'navigation_epic_lookup.json')
with open(LOOKUP_PATH, 'w', encoding='utf-8') as f:
    json.dump(lookup, f, indent=2, ensure_ascii=False)
print(f"Lookup saved: {LOOKUP_PATH}")
print("\nDone!")