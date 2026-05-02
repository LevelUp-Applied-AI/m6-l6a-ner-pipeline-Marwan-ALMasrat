# Stretch 6A-S2 — Multilingual NER Comparison: Analysis

## Overview

This analysis compares Named Entity Recognition performance across English
and Arabic climate texts using two multilingual models:
- **spaCy** `xx_ent_wiki_sm`
- **Hugging Face** `Davlan/xlm-roberta-base-wikiann-ner`

Both models were run on 20 English and 20 Arabic texts from the climate
articles dataset. Label mapping Option B was applied throughout:
`PER → PERSON`, `LOC → LOC`, `ORG → ORG`, `MISC → MISC`.

---

## 1. Comparison Table

| Label   | spaCy-EN | spaCy-AR | HF-EN | HF-AR |
|---------|----------|----------|-------|-------|
| LOC     | 27       | 2        | 27    | 27    |
| MISC    | 15       | 7        | 0     | 0     |
| ORG     | 39       | 2        | 50    | 30    |
| PERSON  | 12       | 9        | 9     | 1     |
| **TOTAL**   | **93**   | **20**   | **86** | **58** |
| density/100w | 7.62 | 2.15 | 7.04 | 6.24 |

> Arabic texts: 930 total words across 20 texts.
> English texts: 1,221 total words across 20 texts.

---

## 2. Zero-Entity Rate

| Combination | Texts with entities | Zero-entity rate |
|-------------|---------------------|-----------------|
| spaCy-EN    | 20/20               | 0.0%            |
| spaCy-AR    | 15/20               | **25.0%**       |
| HF-EN       | 20/20               | 0.0%            |
| HF-AR       | 20/20               | 0.0%            |

spaCy `xx_ent_wiki_sm` failed to find any entity in 5 out of 20 Arabic
texts. HF `xlm-roberta` found at least one entity in every Arabic text,
suggesting stronger Arabic coverage in the transformer-based model.

---

## 3. Paragraph A — Entity Types Harder in Arabic vs English

The comparison table reveals a sharp drop in extraction quality when
moving from English to Arabic. Entity density fell from **7.62 to 2.15
per 100 words** for spaCy, and from **7.04 to 6.24** for HF — a 72%
and 11% drop respectively. The most affected label was **LOC**: spaCy
found 27 location entities in English but only 2 in Arabic, even though
the Arabic texts discuss the same climate topics involving the same
geographic regions. This is directly explained by the absence of
capitalisation in Arabic — spaCy's `xx_ent_wiki_sm` relies heavily on
orthographic cues (capital letters, punctuation patterns) to detect
entity boundaries, and Arabic provides none of these signals. The
qualitative examples confirm the failure mode: spaCy tagged Arabic
words like `ويمثل` ("it represents") and `قدّر` ("estimated") as LOC,
and `وأكد التقرير` ("the report confirmed") as PERSON — these are
clearly false positives caused by the model latching onto arbitrary
token shapes rather than genuine entities. **ORG** showed the same
collapse: 39 English organisations vs only 2 in Arabic for spaCy,
despite texts such as text_id 79 and 84 clearly mentioning the IPCC,
World Bank, and UNDP in Arabic. HF performed substantially better on
Arabic ORG (30 matches vs spaCy's 2), with qualitatively correct
examples such as `الهيئة الحكومية الدولية المعنية بتغير المناخ`
(the IPCC full Arabic name, text_id 79), `البنك الدولي` (World Bank,
text_id 80), and `مؤتمر الأطراف الثامن والعشرين` (COP28, text_id 81).
PERSON entities were the hardest for both models: HF found only 1
Arabic PERSON across 20 texts, and spaCy's 9 apparent matches were
largely wrong — `وأكد وزير` ("the minister confirmed") is a verb
phrase, not a person name. Arabic person names lack the visual
distinctiveness of English proper nouns, and both models clearly
struggle without dedicated Arabic training data.

---

## 4. Paragraph B — Implications for Bilingual NLP in the MENA Region

The results illustrate a fundamental asymmetry that practitioners in
Jordan and the broader MENA region face when building production NLP
systems: the same multilingual model that performs reliably on English
climate text (density 7.04–7.62, zero false-positive PERSON examples)
degrades significantly on Arabic (density 2.15–6.24, multiple
hallucinated entities). For a Jordanian organisation processing
bilingual climate reports — such as the Ministry of Environment
(text_id 4) or UNDP Jordan (text_id 4) — this means an off-the-shelf
multilingual pipeline will silently miss a large fraction of Arabic
named entities while simultaneously producing spurious ones, with no
obvious signal to the downstream consumer that the Arabic output is
less reliable. The practical implication is that bilingual pipelines
in the MENA region require **language-specific quality gates**: the
English branch can rely on general-purpose multilingual models, but
the Arabic branch needs either a dedicated Arabic NER model (such as
`CAMeL-Lab/bert-base-arabic-camelbert-msa-ner`) or a rule-based
pre-processing layer that normalises Arabic morphology and handles
the absence of capitalisation before entity extraction. The 25%
zero-entity rate from spaCy on Arabic texts is itself a deployable
signal — texts where no entities are found could be automatically
routed to a specialised Arabic model rather than returned empty.
HF `xlm-roberta` is the better baseline for Arabic of the two tested,
but its PERSON recall (1 entity across 20 texts) shows it is not
production-ready for Arabic person-name extraction without fine-tuning
on domain-specific Arabic data.

---

## 5. Model Behaviour Summary

| Observation | Evidence |
|-------------|----------|
| spaCy `xx` is unreliable on Arabic LOC | 27 EN vs 2 AR; false positives: `ويمثل`, `قدّر` |
| spaCy `xx` hallucinates Arabic PERSON | `وأكد التقرير`, `وأكد وزير` tagged as PERSON |
| HF `xlm-roberta` handles Arabic ORG well | Correctly found IPCC, World Bank, COP28 in Arabic |
| HF produces `▁` subword artifacts in examples | `▁Antonio ▁Gut er res` — merging works but display is noisy |
| HF misses Arabic PERSON almost entirely | 1 PERSON across 20 Arabic texts |
| spaCy MISC label absent from HF | HF emits no MISC; spaCy emits 15 EN / 7 AR |
| Arabic zero-entity rate | spaCy 25% vs HF 0% — transformer more robust |