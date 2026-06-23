"""
Module 6 Week A — Stretch: Multilingual NER Comparison
Compares NER performance across English and Arabic texts using:
  - spaCy  : xx_ent_wiki_sm  (multilingual)
  - HF     : Davlan/xlm-roberta-base-wikiann-ner  (multilingual)

Label mapping chosen: Option B  (map to English schema where possible)
  PER  -> PERSON
  LOC  -> LOC
  ORG  -> ORG
  MISC -> MISC  (kept as-is, no clean English equivalent)

Run:
    python -m spacy download xx_ent_wiki_sm
    python stretch_multilingual_ner.py
"""

import unicodedata
import pandas as pd
from transformers import pipeline as hf_pipeline
import spacy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_TEXTS   = 20          # minimum texts per language as required
HF_MODEL  = "Davlan/xlm-roberta-base-wikiann-ner"

LABEL_MAP = {           # Option B — multilingual -> English schema
    "PER":  "PERSON",
    "LOC":  "LOC",
    "ORG":  "ORG",
    "MISC": "MISC",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data(filepath="data/climate_articles.csv"):
    df = pd.read_csv(filepath)
    return df


def get_language_split(df, lang, n=N_TEXTS):
    subset = df[df["language"] == lang].head(n).copy()
    print(f"  [{lang}] using {len(subset)} texts")
    return subset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def word_count(text):
    return max(len(str(text).split()), 1)


def entity_density(n_entities, n_words):
    """Entities per 100 words."""
    return round(n_entities / n_words * 100, 2)


def map_label(raw_label):
    """Strip B-/I- prefix then map to English schema."""
    clean = raw_label.split("-", 1)[-1] if "-" in raw_label else raw_label
    return LABEL_MAP.get(clean, clean)


# ---------------------------------------------------------------------------
# spaCy extraction  (xx_ent_wiki_sm)
# ---------------------------------------------------------------------------

def extract_spacy_multilingual(df, nlp):
    rows = []
    zero_entity_texts = 0

    for _, row in df.iterrows():
        text = str(row["text"])
        normalized = unicodedata.normalize("NFC", text)
        doc = nlp(normalized)

        ents = list(doc.ents)
        if not ents:
            zero_entity_texts += 1

        for ent in ents:
            rows.append({
                "text_id":      row["id"],
                "entity_text":  ent.text,
                "entity_label": LABEL_MAP.get(ent.label_, ent.label_),
                "start_char":   ent.start_char,
                "end_char":     ent.end_char,
            })

    print(f"    zero-entity texts: {zero_entity_texts}/{len(df)}")
    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )


# ---------------------------------------------------------------------------
# HF extraction  (xlm-roberta)
# ---------------------------------------------------------------------------

def extract_hf_multilingual(df, ner_pipe):
    rows = []
    zero_entity_texts = 0

    for _, row in df.iterrows():
        text = str(row["text"])

        try:
            raw = ner_pipe(text)
        except Exception as e:
            print(f"    warning: text_id={row['id']} skipped ({e})")
            continue

        # merge subword / continuation tokens
        merged = []
        current_text  = None
        current_label = None
        current_start = None
        current_end   = None

        for token in raw:
            word      = token["word"]
            label_raw = token["entity"]
            start     = token["start"]
            end       = token["end"]

            if label_raw.startswith("I-") and current_text is not None:
                current_text += word[2:] if word.startswith("##") else " " + word
                current_end   = end
                continue

            if current_text is not None:
                merged.append((current_text, current_label, current_start, current_end))

            current_text  = word
            current_label = map_label(label_raw)
            current_start = start
            current_end   = end

        if current_text is not None:
            merged.append((current_text, current_label, current_start, current_end))

        if not merged:
            zero_entity_texts += 1

        for entity_text, entity_label, start_char, end_char in merged:
            rows.append({
                "text_id":      row["id"],
                "entity_text":  entity_text,
                "entity_label": entity_label,
                "start_char":   start_char,
                "end_char":     end_char,
            })

    print(f"    zero-entity texts: {zero_entity_texts}/{len(df)}")
    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )


# ---------------------------------------------------------------------------
# Stats per (language, model) combination
# ---------------------------------------------------------------------------

def compute_stats(entities_df, source_df):
    total_words   = source_df["text"].apply(word_count).sum()
    total_entities = len(entities_df)
    density        = entity_density(total_entities, total_words)
    label_counts   = entities_df["entity_label"].value_counts().to_dict()
    return {
        "total_entities": total_entities,
        "total_words":    total_words,
        "density":        density,
        "label_counts":   label_counts,
    }


def print_stats(label, stats):
    print(f"\n  {label}")
    print(f"    Total entities : {stats['total_entities']}")
    print(f"    Total words    : {stats['total_words']}")
    print(f"    Density        : {stats['density']} per 100 words")
    print(f"    Label breakdown:")
    for lbl, cnt in sorted(stats["label_counts"].items()):
        print(f"      {lbl:<10}: {cnt}")


# ---------------------------------------------------------------------------
# Qualitative examples  (3 per label per combination)
# ---------------------------------------------------------------------------

def print_examples(entities_df, combo_name, n=3):
    print(f"\n  === Examples — {combo_name} ===")
    for label in sorted(entities_df["entity_label"].unique()):
        subset = (
            entities_df[entities_df["entity_label"] == label]
            [["text_id", "entity_text"]]
            .drop_duplicates()
            .head(n)
        )
        print(f"\n  [{label}]")
        print(subset.to_string(index=False))


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results):
    """
    results: dict of  key -> {"stats": stats_dict, "label": str}
    """
    combos = list(results.keys())
    labels = sorted(
        set(
            lbl
            for r in results.values()
            for lbl in r["stats"]["label_counts"]
        )
    )

    # header
    col_w = 14
    header = f"{'Label':<12}" + "".join(f"{c:>{col_w}}" for c in combos)
    print("\n" + "=" * (12 + col_w * len(combos)))
    print("  COMPARISON TABLE — entity counts by label, language, and model")
    print("  Label mapping: Option B (PER→PERSON, LOC→LOC, ORG→ORG, MISC→MISC)")
    print("=" * (12 + col_w * len(combos)))
    print(header)
    print("-" * (12 + col_w * len(combos)))

    for lbl in labels:
        row = f"{lbl:<12}"
        for combo in combos:
            cnt = results[combo]["stats"]["label_counts"].get(lbl, 0)
            row += f"{cnt:>{col_w}}"
        print(row)

    print("-" * (12 + col_w * len(combos)))
    totals = f"{'TOTAL':<12}"
    for combo in combos:
        totals += f"{results[combo]['stats']['total_entities']:>{col_w}}"
    print(totals)

    density_row = f"{'density/100w':<12}"
    for combo in combos:
        density_row += f"{results[combo]['stats']['density']:>{col_w}}"
    print(density_row)
    print("=" * (12 + col_w * len(combos)))

    # column labels legend
    print("\n  Column legend:")
    for combo in combos:
        print(f"    {combo}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Load data ─────────────────────────────────────────
    df = load_data()

    print("\nLoading language subsets...")
    en_df = get_language_split(df, "en", N_TEXTS)
    ar_df = get_language_split(df, "ar", N_TEXTS)

    # ── Load models ───────────────────────────────────────
    print("\nLoading spaCy xx_ent_wiki_sm...")
    nlp_xx = spacy.load("xx_ent_wiki_sm")

    print("Loading HF xlm-roberta...")
    hf_multi = hf_pipeline("ner", model=HF_MODEL)

    # ── Extract entities ──────────────────────────────────
    print("\n--- spaCy xx | English ---")
    spacy_en = extract_spacy_multilingual(en_df, nlp_xx)

    print("--- spaCy xx | Arabic  ---")
    spacy_ar = extract_spacy_multilingual(ar_df, nlp_xx)

    print("--- HF xlm   | English ---")
    hf_en = extract_hf_multilingual(en_df, hf_multi)

    print("--- HF xlm   | Arabic  ---")
    hf_ar = extract_hf_multilingual(ar_df, hf_multi)

    # ── Compute stats ─────────────────────────────────────
    results = {
        "spaCy-EN": {"stats": compute_stats(spacy_en, en_df)},
        "spaCy-AR": {"stats": compute_stats(spacy_ar, ar_df)},
        "HF-EN":    {"stats": compute_stats(hf_en,    en_df)},
        "HF-AR":    {"stats": compute_stats(hf_ar,    ar_df)},
    }

    # ── Print individual stats ────────────────────────────
    for combo, data in results.items():
        print_stats(combo, data["stats"])

    # ── Comparison table ──────────────────────────────────
    print_comparison_table(results)

    # ── Qualitative examples ──────────────────────────────
    print_examples(spacy_en, "spaCy xx — English")
    print_examples(spacy_ar, "spaCy xx — Arabic")
    print_examples(hf_en,    "HF xlm   — English")
    print_examples(hf_ar,    "HF xlm   — Arabic")

    # ── Zero-entity rate summary ──────────────────────────
    print("\n=== Zero-entity rate summary ===")
    for combo, src_df, ent_df in [
        ("spaCy-EN", en_df, spacy_en),
        ("spaCy-AR", ar_df, spacy_ar),
        ("HF-EN",    en_df, hf_en),
        ("HF-AR",    ar_df, hf_ar),
    ]:
        texts_with_ents = ent_df["text_id"].nunique()
        total_texts     = len(src_df)
        zero_rate       = round((total_texts - texts_with_ents) / total_texts * 100, 1)
        print(f"  {combo:<12}: {texts_with_ents}/{total_texts} texts had entities "
              f"({zero_rate}% zero-entity rate)")

    # ── Note on evaluation ────────────────────────────────
    print("\n=== Evaluation Note ===")
    print("  English precision/recall/F1 not computed here.")
    print("  Use evaluate_ner() from ner_pipeline.py with the gold standard.")
    print("  Arabic: no gold annotations available — evaluation is qualitative only.")