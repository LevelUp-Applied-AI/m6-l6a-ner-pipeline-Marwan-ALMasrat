"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import unicodedata

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# Task 1 — Load & Explore
# ---------------------------------------------------------------------------

def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    # Word count per row (split on whitespace)
    word_counts = df["text"].dropna().apply(lambda t: len(t.split()))

    return {
        "shape": df.shape,
        "lang_counts": df["language"].value_counts().to_dict(),
        "category_counts": df["category"].value_counts().to_dict(),
        "text_length_stats": {
            "mean": round(word_counts.mean(), 2),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
        },
    }


# ---------------------------------------------------------------------------
# Task 2 — Preprocess Text
# ---------------------------------------------------------------------------

def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # NFC Unicode normalization
    normalized = unicodedata.normalize("NFC", text)

    # Run through spaCy
    doc = nlp(normalized)

    # Drop punctuation and whitespace tokens; return lowercased lemmas
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space
    ]
    return tokens


# ---------------------------------------------------------------------------
# Task 3 — spaCy NER
# ---------------------------------------------------------------------------

def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    en_df = df[df["language"] == "en"].copy()
    rows = []

    for _, row in en_df.iterrows():
        doc = nlp(row["text"])
        for ent in doc.ents:
            rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return pd.DataFrame(rows, columns=["text_id", "entity_text", "entity_label",
                                        "start_char", "end_char"])


# ---------------------------------------------------------------------------
# Task 4 — Hugging Face NER
# ---------------------------------------------------------------------------

def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Steps:
      1. Filter to English rows.
      2. Run each text through the HF pipeline.
      3. Merge ## subword (WordPiece) continuation tokens.
      4. Strip the IOB B-/I- prefix from labels.

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    en_df = df[df["language"] == "en"].copy()
    rows = []

    for _, row in en_df.iterrows():
        raw_entities = ner_pipeline(row["text"])

        # --- Merge subword tokens and build clean entity spans ---
        merged = []          # list of (entity_text, label, start, end)
        current_text = None
        current_label = None
        current_start = None
        current_end = None

        for token in raw_entities:
            word = token["word"]
            label_raw = token["entity"]  # e.g. "B-ORG", "I-ORG"
            start = token["start"]
            end = token["end"]

            # Continuation subword token (##)
            if label_raw.startswith("I-") and current_text is not None:
                if word.startswith("##"):
                  current_text += word[2:]   # بدون مسافة
                else:
                  current_text += " " + word # بمسافة
                current_end = end
                continue
            # New token — flush the previous entity if any
            if current_text is not None:
                merged.append((current_text, current_label, current_start, current_end))

            # Strip IOB prefix: "B-ORG" -> "ORG", "I-ORG" -> "ORG"
            label_clean = label_raw.split("-", 1)[-1] if "-" in label_raw else label_raw

            current_text = word
            current_label = label_clean
            current_start = start
            current_end = end

        # Flush the last entity
        if current_text is not None:
            merged.append((current_text, current_label, current_start, current_end))

        # Build rows from merged entities
        for entity_text, entity_label, start_char, end_char in merged:
            rows.append({
                "text_id": row["id"],
                "entity_text": entity_text,
                "entity_label": entity_label,
                "start_char": start_char,
                "end_char": end_char,
            })

    return pd.DataFrame(rows, columns=["text_id", "entity_text", "entity_label",
                                        "start_char", "end_char"])


# ---------------------------------------------------------------------------
# Task 5 — Compare NER Outputs
# ---------------------------------------------------------------------------

def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    spacy_counts = spacy_df["entity_label"].value_counts().to_dict()
    hf_counts    = hf_df["entity_label"].value_counts().to_dict()
    # خطوة 1: lowercase للمقارنة
    spacy_lower = set(zip(spacy_df["text_id"], spacy_df["entity_text"].str.lower()))
    hf_lower    = set(zip(hf_df["text_id"],    hf_df["entity_text"].str.lower()))

# خطوة 2: الأزواج الأصلية
    spacy_pairs = set(zip(spacy_df["text_id"], spacy_df["entity_text"]))
    hf_pairs    = set(zip(hf_df["text_id"],    hf_df["entity_text"]))

# خطوة 3: مقارنة عبر lowercase
    both_lower = spacy_lower & hf_lower
    both       = {p for p in spacy_pairs if (p[0], p[1].lower()) in both_lower}
    spacy_only = {p for p in spacy_pairs if (p[0], p[1].lower()) not in hf_lower}
    hf_only    = {p for p in hf_pairs    if (p[0], p[1].lower()) not in spacy_lower}
    
    result = {
        "spacy_counts": spacy_counts,
        "hf_counts":    hf_counts,
        "total_spacy":  len(spacy_df),
        "total_hf":     len(hf_df),
        "both":         both,
        "spacy_only":   spacy_only,
        "hf_only":      hf_only,
    }

    # Human-readable summary
    print("\n=== NER Comparison Summary ===")
    print(f"Total spaCy entities : {result['total_spacy']}")
    print(f"Total HF entities    : {result['total_hf']}")
    print(f"Agreed on            : {len(both)}")
    print(f"spaCy-only           : {len(spacy_only)}")
    print(f"HF-only              : {len(hf_only)}")
    print("\nspaCy entity-type counts:")
    for label, count in sorted(spacy_counts.items()):
        print(f"  {label:10s}: {count}")
    print("\nHF entity-type counts:")
    for label, count in sorted(hf_counts.items()):
        print(f"  {label:10s}: {count}")

    return result


# ---------------------------------------------------------------------------
# Task 6 — Evaluate Against Gold Standard
# ---------------------------------------------------------------------------

def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    An entity is a true positive only when text_id, entity_text, AND
    entity_label all match a gold entry.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text, entity_label.
        gold_df:      DataFrame with columns text_id, entity_text, entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    # Work only with the texts that have gold annotations
    gold_text_ids = set(gold_df["text_id"].unique())
    pred_filtered = predicted_df[predicted_df["text_id"].isin(gold_text_ids)].copy()

    # Represent each entity as a (text_id, entity_text, entity_label) tuple
    gold_set = set(
        zip(gold_df["text_id"], gold_df["entity_text"], gold_df["entity_label"])
    )
    pred_set = set(
        zip(pred_filtered["text_id"], pred_filtered["entity_text"],
            pred_filtered["entity_label"])
    )

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp    = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # ── Task 1: Load & Explore ──────────────────────────────────────────────
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape            : {summary['shape']}")
            print(f"Languages        : {summary['lang_counts']}")
            print(f"Categories       : {summary['category_counts']}")
            print(f"Text length (w)  : {summary['text_length_stats']}")

        # ── Task 2: Preprocess sample ───────────────────────────────────────
        sample_row    = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # ── Task 3: spaCy NER ──────────────────────────────────────────────
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")
            print(spacy_entities.head())

        # ── Task 4: HF NER ─────────────────────────────────────────────────
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"\nHF entities: {len(hf_entities)} total")
            print(hf_entities.head())

        # ── Task 5: Compare ────────────────────────────────────────────────
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only : {len(comparison['spacy_only'])}")
                print(f"HF-only    : {len(comparison['hf_only'])}")

        # ── Task 6: Evaluate ───────────────────────────────────────────────
        gold = pd.read_csv("data/gold_entities.csv")

        if spacy_entities is not None:
            spacy_metrics = evaluate_ner(spacy_entities, gold)
            if spacy_metrics is not None:
                print(f"\nspaCy evaluation : {spacy_metrics}")

        if hf_entities is not None:
            hf_metrics = evaluate_ner(hf_entities, gold)
            if hf_metrics is not None:
                print(f"HF evaluation    : {hf_metrics}")