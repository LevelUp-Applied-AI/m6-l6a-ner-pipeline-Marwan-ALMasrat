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
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
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
    normalized = unicodedata.normalize("NFC", text)
    doc = nlp(normalized)
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
    en_df = df[df["language"] == "en"].copy()
    rows = []

    for _, row in en_df.iterrows():
        raw_entities = ner_pipeline(row["text"])

        merged = []
        current_text = None
        current_label = None
        current_start = None
        current_end = None

        for token in raw_entities:
            word = token["word"]
            label_raw = token["entity"]
            start = token["start"]
            end = token["end"]
            

            if label_raw.startswith("I-") and current_text is not None:
                if word.startswith("##"):
                    current_text += word[2:]
                else:
                    current_text += " " + word
                current_end = end
                continue


            if current_text is not None:
                merged.append((current_text, current_label, current_start, current_end))

            label_clean = label_raw.split("-", 1)[-1] if "-" in label_raw else label_raw

            current_text = word
            current_label = label_clean
            current_start = start
            current_end = end

        if current_text is not None:
            merged.append((current_text, current_label, current_start, current_end))

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
    spacy_counts = spacy_df["entity_label"].value_counts().to_dict()
    hf_counts    = hf_df["entity_label"].value_counts().to_dict()

    # lowercase للمقارنة
    spacy_lower = set(zip(spacy_df["text_id"], spacy_df["entity_text"].str.lower()))
    hf_lower    = set(zip(hf_df["text_id"],    hf_df["entity_text"].str.lower()))

    # النص الأصلي للنتائج
    spacy_pairs = set(zip(spacy_df["text_id"], spacy_df["entity_text"]))
    hf_pairs    = set(zip(hf_df["text_id"],    hf_df["entity_text"]))

    both_lower  = spacy_lower & hf_lower

    both        = {p for p in spacy_pairs if (p[0], p[1].lower()) in both_lower}
    spacy_only  = {p for p in spacy_pairs if (p[0], p[1].lower()) not in hf_lower}
    hf_only     = {p for p in hf_pairs    if (p[0], p[1].lower()) not in spacy_lower}

    result = {
        "spacy_counts": spacy_counts,
        "hf_counts":    hf_counts,
        "total_spacy":  len(spacy_df),
        "total_hf":     len(hf_df),
        "both":         both,
        "spacy_only":   spacy_only,
        "hf_only":      hf_only,
    }

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
    gold_text_ids = set(gold_df["text_id"].unique())
    pred_filtered = predicted_df[predicted_df["text_id"].isin(gold_text_ids)].copy()

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

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nlp    = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape            : {summary['shape']}")
            print(f"Languages        : {summary['lang_counts']}")
            print(f"Categories       : {summary['category_counts']}")
            print(f"Text length (w)  : {summary['text_length_stats']}")

        sample_row    = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")
            print(spacy_entities.head())

        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"\nHF entities: {len(hf_entities)} total")
            print(hf_entities.head())

        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only : {len(comparison['spacy_only'])}")
                print(f"HF-only    : {len(comparison['hf_only'])}")

        gold = pd.read_csv("data/gold_entities.csv")

        if spacy_entities is not None:
            spacy_metrics = evaluate_ner(spacy_entities, gold)
            if spacy_metrics is not None:
                print(f"\nspaCy evaluation : {spacy_metrics}")

        if hf_entities is not None:
            hf_metrics = evaluate_ner(hf_entities, gold)
            if hf_metrics is not None:
                print(f"HF evaluation    : {hf_metrics}")