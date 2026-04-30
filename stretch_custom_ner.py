"""
Module 6 Week A — Stretch: Custom NER Rules
Extends the base spaCy NER with domain-specific EntityRuler patterns
for climate terminology.

Run: python stretch_custom_ner.py
"""

import pandas as pd
import spacy
from ner_pipeline import load_data, evaluate_ner, extract_spacy_entities


# ---------------------------------------------------------------------------
# Entity Ruler Patterns
# ---------------------------------------------------------------------------

PATTERNS = [
    # CLIMATE_EVENT — 3 distinct concepts
    {"label": "CLIMATE_EVENT", "pattern": "COP28"},
    {"label": "CLIMATE_EVENT", "pattern": "COP27"},
    {"label": "CLIMATE_EVENT", "pattern": "Bonn Climate Change Conference"},
    {"label": "CLIMATE_EVENT", "pattern": "Climate Ambition Summit"},
    {"label": "CLIMATE_EVENT", "pattern": "UN Climate Conference"},

    # POLICY — 3 distinct concepts
    {"label": "POLICY", "pattern": "Paris Agreement"},
    {"label": "POLICY", "pattern": "Carbon Border Adjustment Mechanism"},
    {"label": "POLICY", "pattern": "CBAM"},

    # REPORT — 2 distinct concepts  (two entries for the same report = allowed)
    {"label": "REPORT", "pattern": "Sixth Assessment Report"},
    {"label": "REPORT", "pattern": "IPCC AR6"},

    # THRESHOLD — 3 distinct concepts
    {"label": "THRESHOLD", "pattern": "1.5 degrees Celsius"},
    {"label": "THRESHOLD", "pattern": [{"LOWER": "net"}, {"LOWER": "zero"}]},
    {"label": "THRESHOLD", "pattern": [{"TEXT": {"REGEX": r"^[12]\.?\d?°?C$"}}]},

    # AGREEMENT — 1 distinct concept (second entry for same = allowed, covers 8 distinct)
    {"label": "AGREEMENT", "pattern": "Kyoto Protocol"},
]

# Total: 14 entries, covers 11 distinct concepts  →  satisfies constraint


# ---------------------------------------------------------------------------
# Build pipelines
# ---------------------------------------------------------------------------

def build_pipeline_before(patterns):
    """EntityRuler inserted BEFORE the statistical NER — ruler takes priority."""
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)
    return nlp


def build_pipeline_after(patterns):
    """EntityRuler inserted AFTER the statistical NER — NER takes priority."""
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", after="ner")
    ruler.add_patterns(patterns)
    return nlp


# ---------------------------------------------------------------------------
# Extract entities with custom ruler
# ---------------------------------------------------------------------------

def extract_custom_entities(df, nlp):
    """Same extraction logic as base lab, works with any nlp pipeline."""
    en_df = df[df["language"] == "en"].copy()
    rows = []
    for _, row in en_df.iterrows():
        doc = nlp(row["text"])
        for ent in doc.ents:
            rows.append({
                "text_id":      row["id"],
                "entity_text":  ent.text,
                "entity_label": ent.label_,
                "start_char":   ent.start_char,
                "end_char":     ent.end_char,
            })
    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )


# ---------------------------------------------------------------------------
# Evaluation — standard labels only (as required by the assignment)
# ---------------------------------------------------------------------------

STANDARD_LABELS = {
    "ORG", "GPE", "DATE", "LAW", "MONEY",
    "PERSON", "QUANTITY", "LOC", "EVENT", "WORK_OF_ART",
}


def evaluate_standard_only(predicted_df, gold_df):
    """
    Filter to STANDARD_LABELS before evaluation.
    Custom labels (CLIMATE_EVENT, POLICY, REPORT, THRESHOLD, AGREEMENT)
    are excluded to avoid artificially depressing precision.
    """
    pred_std = predicted_df[predicted_df["entity_label"].isin(STANDARD_LABELS)].copy()
    return evaluate_ner(pred_std, gold_df)


# ---------------------------------------------------------------------------
# Qualitative check — where did custom rules fire?
# ---------------------------------------------------------------------------

def custom_label_examples(entities_df, n=5):
    """
    Return example rows for each custom label so we can assess
    whether the rule fired correctly.
    """
    custom_labels = set(entities_df["entity_label"]) - STANDARD_LABELS
    examples = {}
    for label in sorted(custom_labels):
        subset = entities_df[entities_df["entity_label"] == label]
        examples[label] = subset[["text_id", "entity_text"]].drop_duplicates().head(n)
    return examples


# ---------------------------------------------------------------------------
# Compare entity counts before and after adding ruler
# ---------------------------------------------------------------------------

def compare_entity_counts(base_df, before_df, after_df):
    base_counts   = base_df["entity_label"].value_counts().to_dict()
    before_counts = before_df["entity_label"].value_counts().to_dict()
    after_counts  = after_df["entity_label"].value_counts().to_dict()

    all_labels = sorted(
        set(base_counts) | set(before_counts) | set(after_counts)
    )

    print("\n=== Entity Count Comparison ===")
    print(f"{'Label':<22} {'Base':>8} {'Ruler-Before':>14} {'Ruler-After':>13}")
    print("-" * 60)
    for label in all_labels:
        b  = base_counts.get(label, 0)
        rb = before_counts.get(label, 0)
        ra = after_counts.get(label, 0)
        print(f"{label:<22} {b:>8} {rb:>14} {ra:>13}")
    print("-" * 60)
    print(f"{'TOTAL':<22} {sum(base_counts.values()):>8} "
          f"{sum(before_counts.values()):>14} {sum(after_counts.values()):>13}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Load data & gold standard ─────────────────────────
    df   = load_data()
    gold = pd.read_csv("data/gold_entities.csv")

    # ── Base spaCy (no ruler) ─────────────────────────────
    nlp_base = spacy.load("en_core_web_sm")
    base_entities = extract_custom_entities(df, nlp_base)
    base_metrics  = evaluate_standard_only(base_entities, gold)

    print("=" * 60)
    print("  BASE spaCy (no EntityRuler)")
    print("=" * 60)
    print(f"  Total entities : {len(base_entities)}")
    print(f"  Precision      : {base_metrics['precision']}")
    print(f"  Recall         : {base_metrics['recall']}")
    print(f"  F1             : {base_metrics['f1']}")

    # ── Ruler BEFORE NER ─────────────────────────────────
    nlp_before      = build_pipeline_before(PATTERNS)
    before_entities = extract_custom_entities(df, nlp_before)
    before_metrics  = evaluate_standard_only(before_entities, gold)

    print("\n" + "=" * 60)
    print("  EntityRuler BEFORE NER")
    print("=" * 60)
    print(f"  Total entities : {len(before_entities)}")
    print(f"  Precision      : {before_metrics['precision']}")
    print(f"  Recall         : {before_metrics['recall']}")
    print(f"  F1             : {before_metrics['f1']}")

    # ── Ruler AFTER NER ──────────────────────────────────
    nlp_after      = build_pipeline_after(PATTERNS)
    after_entities = extract_custom_entities(df, nlp_after)
    after_metrics  = evaluate_standard_only(after_entities, gold)

    print("\n" + "=" * 60)
    print("  EntityRuler AFTER NER")
    print("=" * 60)
    print(f"  Total entities : {len(after_entities)}")
    print(f"  Precision      : {after_metrics['precision']}")
    print(f"  Recall         : {after_metrics['recall']}")
    print(f"  F1             : {after_metrics['f1']}")

    # ── Side-by-side metrics delta ────────────────────────
    print("\n" + "=" * 60)
    print("  Metrics Delta (vs base)")
    print("=" * 60)
    print(f"  {'Metric':<12} {'Base':>8} {'Before':>8} {'After':>8}")
    print(f"  {'-'*38}")
    for key in ["precision", "recall", "f1"]:
        print(
            f"  {key:<12} "
            f"{base_metrics[key]:>8.4f} "
            f"{before_metrics[key]:>8.4f} "
            f"{after_metrics[key]:>8.4f}"
        )

    # ── Entity count comparison ───────────────────────────
    compare_entity_counts(base_entities, before_entities, after_entities)

    # ── Qualitative: where did custom rules fire? ─────────
    print("\n=== Custom Label Examples — Ruler BEFORE ===")
    for label, examples_df in custom_label_examples(before_entities).items():
        print(f"\n  [{label}]")
        print(examples_df.to_string(index=False))

    print("\n=== Custom Label Examples — Ruler AFTER ===")
    for label, examples_df in custom_label_examples(after_entities).items():
        print(f"\n  [{label}]")
        print(examples_df.to_string(index=False))

    # ── Pipeline position difference ─────────────────────
    print("\n=== Pipeline Position: Before vs After ===")
    print("  BEFORE: EntityRuler matches override the statistical NER.")
    print("          Useful when you trust your patterns more than the model.")
    print("  AFTER:  Statistical NER matches take priority; ruler fills gaps.")
    print("          Safer when base NER is reliable but misses domain terms.")
    before_custom = before_entities[
        ~before_entities["entity_label"].isin(STANDARD_LABELS)
    ]
    after_custom = after_entities[
        ~after_entities["entity_label"].isin(STANDARD_LABELS)
    ]
    print(f"\n  Custom-label entities  BEFORE position: {len(before_custom)}")
    print(f"  Custom-label entities  AFTER  position: {len(after_custom)}")