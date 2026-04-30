# Stretch 6A-S1 — Custom NER Rules: Analysis

## Overview

This analysis documents the impact of adding a domain-specific `EntityRuler`
to the base spaCy NER pipeline on a climate-related text corpus.
Fourteen patterns were defined across five custom labels:
`CLIMATE_EVENT`, `POLICY`, `REPORT`, `THRESHOLD`, and `AGREEMENT`.
The ruler was tested in two pipeline positions — before and after the
statistical NER — and evaluated on the gold-standard subset
(`gold_entities.csv`) using standard labels only.

---

## 1. Metrics Summary

| Metric    | Base spaCy | Ruler BEFORE | Ruler AFTER |
|-----------|-----------|--------------|-------------|
| Precision | 0.6567    | **0.7458**   | 0.6567      |
| Recall    | 0.6471    | 0.6471       | 0.6471      |
| F1        | 0.6519    | **0.6929**   | 0.6519      |

> Evaluation is restricted to standard spaCy labels only
> (`ORG`, `GPE`, `DATE`, `LAW`, `MONEY`, `PERSON`, `QUANTITY`,
> `LOC`, `EVENT`, `WORK_OF_ART`) to avoid artificially depressing
> Precision with custom labels that cannot match gold entries.

---

## 2. Pipeline Position: Before vs After

### Ruler BEFORE NER

When the `EntityRuler` runs before the statistical NER, its matches
take priority and **lock in** the span before the model sees it.
This caused several base-NER misclassifications to be corrected:

- **`COP28`** was previously tagged as `ORG` by the base model
  (text_id 2, 10). The ruler correctly relabelled it `CLIMATE_EVENT`,
  removing a spurious `ORG` match and improving Precision from
  0.6567 to **0.7458**.
- **`Bonn Climate Change Conference`** (text_id 5) and
  **`Climate Ambition Summit`** (text_id 7) were missed or
  misclassified by the base model; the ruler captured both correctly
  as `CLIMATE_EVENT`.
- **`Paris Agreement`** appeared in text_ids 5, 6, 7, and 10 — the
  base model labelled it inconsistently (`LAW` or missed). The ruler
  consistently tagged it `POLICY`.
- **`Sixth Assessment Report`** (text_id 1) was missed by base NER
  entirely; the ruler captured it as `REPORT`.
- **`1.5 degrees Celsius`** (text_id 1) and **`Net Zero`** (text_id 50)
  were both correctly tagged as `THRESHOLD`.

Total entity count increased from **1202 → 1206**.
Custom-label entities added: **310** (includes all CARDINAL, PERCENT,
etc. that appear in the before-position output — the meaningful
custom additions are the 17 new domain-label matches).

### Ruler AFTER NER

When the ruler runs after the statistical NER, the model's spans
take priority and the ruler only fills genuine gaps.
The result was **no change** in standard-label Precision or Recall
(both identical to base: P=0.6567, R=0.6471, F1=0.6519).

The key observation from the entity count table:

| Label         | Base | Before | After |
|---------------|------|--------|-------|
| CLIMATE_EVENT | 0    | 7      | 3     |
| POLICY        | 0    | 7      | 0     |
| REPORT        | 0    | 1      | 0     |
| THRESHOLD     | 0    | 2      | 0     |
| EVENT         | 8    | 2      | 8     |

In the **after** position, the statistical NER had already tagged
`COP28` as `ORG` and `Paris Agreement` as `LAW`, so the ruler's
patterns were blocked. Only 3 `CLIMATE_EVENT` matches appeared
(the cases where the base model had no overlapping span), and
`POLICY`, `REPORT`, and `THRESHOLD` returned zero matches entirely.
This confirms that for well-known named entities, the base model
tends to assign its own label first, making the **after** position
ineffective for relabelling.

**Conclusion on position:** `before` is the correct choice for
domain-specific relabelling of entities the base model
misclassifies. `after` is only useful for genuinely novel spans
the model produces no output for.

---

## 3. Where Custom Rules Helped

### CLIMATE_EVENT
- `COP28` correctly identified in text_ids 2, 10, and 48.
  Previously classified as `ORG` by base model, which is technically
  defensible but misleading in a climate-domain context.
- `Bonn Climate Change Conference` (text_id 5) and
  `Climate Ambition Summit` (text_id 7) were entirely missed by the
  base model and correctly captured by the ruler.

### POLICY
- `Paris Agreement` appeared 4 times across the gold subset
  (text_ids 5, 6, 7, 10). The base model labelled it `LAW` in some
  texts and missed it in others. The custom `POLICY` label
  provides a more semantically precise classification for a
  climate-domain pipeline.
- `Carbon Border Adjustment Mechanism` (text_id 6) was captured
  correctly; the base model had no match for this entity.

### REPORT
- `Sixth Assessment Report` (text_id 1) was missed by the base
  model. The ruler correctly identified it, which is meaningful
  because this is the primary IPCC scientific reference document
  in the dataset.

### THRESHOLD
- `1.5 degrees Celsius` (text_id 1) was previously tagged
  `QUANTITY` by the base model — technically correct but loses
  the climate significance. The custom `THRESHOLD` label
  preserves domain meaning.
- `Net Zero` (text_id 50) was missed entirely by the base model
  and correctly captured by the ruler.

---

## 4. Where Custom Rules Introduced Noise

### EVENT → CLIMATE_EVENT conflict (before position)
The `EVENT` count dropped from **8 → 2** when the ruler ran before
the NER. This is because some spans the base model would have
tagged `EVENT` were intercepted and relabelled `CLIMATE_EVENT`
by the ruler. Since `EVENT` is a standard gold label and
`CLIMATE_EVENT` is not, this conversion caused those spans to
disappear from the standard-label evaluation — they became
invisible to the gold comparison. This is the primary precision
gain mechanism: fewer incorrect `ORG`/`EVENT` spans in the
standard-label set means fewer FP.

### ORG → custom label conversion
`ORG` count dropped from **184 → 180** in the before position,
and `GPE` from **165 → 164**. These are cases where the ruler's
spans overlapped with what the base model would have produced,
and the ruler's label won. In most cases this is correct
(`COP28` should not be `ORG`), but it is worth noting that
some legitimate `ORG` spans may have been suppressed if a
pattern partially overlapped.

### No false positives from phrase patterns
The phrase patterns (`"Paris Agreement"`, `"COP28"`, etc.) showed
no false positives in the dataset — all fired instances were
genuinely the intended entity. The token pattern for temperature
thresholds (`REGEX: ^[12]\.?\d?°?C$`) produced no matches in
this corpus, suggesting the dataset uses expanded forms
(`"1.5 degrees Celsius"`) rather than abbreviated notation.

---

## 5. Summary

Adding a domain-specific `EntityRuler` in the **before** position
improved Precision by **+8.9 percentage points** (0.6567 → 0.7458)
and F1 by **+4.1 points** (0.6519 → 0.6929) with no change to
Recall. The gain came from two mechanisms:

1. **Relabelling** — entities the base model tagged with wrong
   standard labels (e.g., `COP28` as `ORG`) were intercepted and
   given custom labels, removing them from the FP pool.
2. **Gap filling** — entities the base model missed entirely
   (`Sixth Assessment Report`, `Net Zero`, `Bonn Climate Change
   Conference`) were captured.

Recall remained unchanged (0.6471) because the gold standard
contains only standard labels, and the new custom-label matches
cannot contribute to TP in the standard-label evaluation. In a
production system with a gold standard that includes custom labels,
Recall would also improve.

The **after** position produced no measurable improvement because
the statistical NER's spans dominated, blocking the ruler from
relabelling misclassified entities. For domain relabelling tasks,
the **before** position is the correct engineering choice.