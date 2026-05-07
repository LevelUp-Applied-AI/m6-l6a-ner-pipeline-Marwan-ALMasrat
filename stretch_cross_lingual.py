"""
Module 6 Week B — Stretch: Cross-Lingual Embedding Comparison
stretch_cross_lingual.py

Extracts multilingual BERT embeddings for English and Arabic climate texts,
computes a 20x20 cosine similarity matrix, and visualizes a heatmap.

Run: python stretch_cross_lingual.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Arabic text rendering fix
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    print("⚠️  Install arabic-reshaper and python-bidi for correct Arabic display:")
    print("    pip install arabic-reshaper python-bidi")


def fix_arabic(text, max_chars=40):
    """Reshape and reorder Arabic text for correct RTL rendering in matplotlib."""
    text = str(text).strip().replace("\n", " ")[:max_chars]
    if ARABIC_SUPPORT:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    return text


# ---------------------------------------------------------------------------
# 1 — Load & Filter Data
# ---------------------------------------------------------------------------

def load_bilingual_data(filepath="data/climate_articles.csv", n=10):
    """
    Load the climate dataset and select n English + n Arabic texts.
    Where possible, prefer texts that cover matching topics across languages.
    """
    df = pd.read_csv(filepath)

    en_df = df[df["language"] == "en"].dropna(subset=["text"]).head(n).reset_index(drop=True)
    ar_df = df[df["language"] == "ar"].dropna(subset=["text"]).head(n).reset_index(drop=True)

    print(f"Selected {len(en_df)} English texts and {len(ar_df)} Arabic texts.")
    return en_df, ar_df


# ---------------------------------------------------------------------------
# 2 — Load Multilingual BERT
# ---------------------------------------------------------------------------

def load_model(model_name="bert-base-multilingual-cased"):
    """
    Load the multilingual BERT tokenizer and model.
    ~680MB download on first run.
    """
    print(f"\nLoading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# 3 — Extract Embeddings (mean pooling over last hidden states)
# ---------------------------------------------------------------------------

def mean_pooling(model_output, attention_mask):
    """
    Mean pool the token embeddings, ignoring padding tokens.
    """
    token_embeddings = model_output.last_hidden_state          # (batch, seq_len, hidden)
    mask_expanded = attention_mask.unsqueeze(-1).float()       # (batch, seq_len, 1)
    summed = (token_embeddings * mask_expanded).sum(dim=1)     # (batch, hidden)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)          # (batch, 1)
    return (summed / counts).detach().numpy()                  # (batch, hidden)


def extract_embeddings(texts, tokenizer, model, max_length=512):
    """
    Extract one embedding vector per text using mean pooling.
    Returns a numpy array of shape (n_texts, hidden_size).
    """
    embeddings = []
    for i, text in enumerate(texts):
        print(f"  Embedding {i+1}/{len(texts)} ...", end="\r")
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        with torch.no_grad():
            output = model(**encoded)
        emb = mean_pooling(output, encoded["attention_mask"])  # (1, hidden)
        embeddings.append(emb[0])
    print()
    return np.array(embeddings)  # (n_texts, hidden)


# ---------------------------------------------------------------------------
# 4 — Compute Cosine Similarity Matrix
# ---------------------------------------------------------------------------

def compute_similarity_matrix(en_embeddings, ar_embeddings):
    """
    Compute a (n_en x n_ar) cosine similarity matrix.
    """
    sim_matrix = cosine_similarity(en_embeddings, ar_embeddings)
    return sim_matrix


# ---------------------------------------------------------------------------
# 5 — Heatmap Visualization
# ---------------------------------------------------------------------------

def truncate_label(text, max_chars=40):
    text = str(text).strip().replace("\n", " ")
    return text[:max_chars] + "…" if len(text) > max_chars else text


def plot_heatmap(sim_matrix, en_texts, ar_texts, output_path="stretch_heatmap.png"):
    """
    Plot and save a heatmap of the cosine similarity matrix.
    Rows = English texts, Columns = Arabic texts.
    """
    en_labels = [truncate_label(t) for t in en_texts]
    ar_labels = [fix_arabic(t, max_chars=40) for t in ar_texts]

    fig, ax = plt.subplots(figsize=(16, 10))

    sns.heatmap(
        sim_matrix,
        xticklabels=ar_labels,
        yticklabels=en_labels,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        ax=ax,
    )

    ax.set_title(
        "Cross-Lingual Cosine Similarity\n(bert-base-multilingual-cased)",
        fontsize=14,
        pad=14,
    )
    ax.set_xlabel("Arabic Texts (first 40 chars)", fontsize=11)
    ax.set_ylabel("English Texts (first 40 chars)", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to: {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 6 — Analysis Helper: Cross-lingual vs Within-language
# ---------------------------------------------------------------------------

def print_similarity_analysis(sim_matrix, en_df, ar_df, en_embeddings, ar_embeddings):
    """
    Print cross-lingual similarity stats and compare to within-language similarity.
    """
    print("\n=== Cross-Lingual Similarity Analysis ===")

    # Top 5 most similar English–Arabic pairs
    flat = [(sim_matrix[i, j], i, j)
            for i in range(sim_matrix.shape[0])
            for j in range(sim_matrix.shape[1])]
    flat.sort(reverse=True)

    print("\nTop 5 most similar English–Arabic pairs:")
    for score, i, j in flat[:5]:
        en_snippet = truncate_label(en_df.iloc[i]["text"], 60)
        ar_snippet = truncate_label(ar_df.iloc[j]["text"], 60)
        print(f"  Score={score:.4f} | EN: {en_snippet}")
        print(f"           | AR: {ar_snippet}\n")

    # Bottom 3 pairs
    print("Bottom 3 (least similar) English–Arabic pairs:")
    for score, i, j in flat[-3:]:
        en_snippet = truncate_label(en_df.iloc[i]["text"], 60)
        ar_snippet = truncate_label(ar_df.iloc[j]["text"], 60)
        print(f"  Score={score:.4f} | EN: {en_snippet}")
        print(f"           | AR: {ar_snippet}\n")

    # Within-language similarity (English only)
    en_within = cosine_similarity(en_embeddings)
    np.fill_diagonal(en_within, np.nan)
    en_mean = np.nanmean(en_within)

    cross_mean = sim_matrix.mean()

    print(f"Mean cross-lingual similarity (EN–AR) : {cross_mean:.4f}")
    print(f"Mean within-language similarity (EN–EN): {en_mean:.4f}")
    print(
        f"\n→ Cross-lingual scores are "
        f"{'lower' if cross_mean < en_mean else 'higher'} than "
        f"within-language scores by {abs(en_mean - cross_mean):.4f} points."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load data
    en_df, ar_df = load_bilingual_data("data/climate_articles.csv", n=10)

    # 2. Load model
    tokenizer, model = load_model("bert-base-multilingual-cased")

    # 3. Extract embeddings
    print("\nExtracting English embeddings ...")
    en_embeddings = extract_embeddings(en_df["text"].tolist(), tokenizer, model)

    print("Extracting Arabic embeddings ...")
    ar_embeddings = extract_embeddings(ar_df["text"].tolist(), tokenizer, model)

    print(f"\nEmbedding shapes — EN: {en_embeddings.shape}, AR: {ar_embeddings.shape}")

    # 4. Cosine similarity matrix
    sim_matrix = compute_similarity_matrix(en_embeddings, ar_embeddings)
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    # 5. Heatmap
    plot_heatmap(sim_matrix, en_df["text"], ar_df["text"],
                 output_path="stretch_heatmap.png")

    # 6. Analysis
    print_similarity_analysis(sim_matrix, en_df, ar_df, en_embeddings, ar_embeddings)