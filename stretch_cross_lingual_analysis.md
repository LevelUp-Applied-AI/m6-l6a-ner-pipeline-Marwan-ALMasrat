# Stretch 6B-S2 — Cross-Lingual Embedding Analysis

## Part A: Cross-Lingual Similarity Quality

The `bert-base-multilingual-cased` model showed a clear ability to capture semantic similarity
across English and Arabic climate texts, though with lower absolute scores than within-language
comparisons. The highest-scoring pair in the matrix was the IPCC Sixth Assessment Report
English text paired with an Arabic text about international climate agreements (score = **0.75**),
followed closely by the same IPCC text paired with Arabic text about the Jordanian climate
policy (score = **0.74**). These high scores make sense: all three texts revolve around
multilateral climate frameworks and policy language, which the multilingual model has
learned to place in nearby regions of the shared embedding space. On the other end, the
lowest score observed was **0.48**, between the United Nations General Assembly text (English)
and an Arabic text about carbon studies — two texts that share the climate domain but differ
substantially in subject matter. Importantly, the ranking of pairs is preserved: same-topic
cross-lingual pairs consistently scored higher than off-topic pairs, which is the key
indicator that the shared multilingual space is functioning correctly even if absolute scores
(range ≈ 0.48–0.75) are lower than typical within-language English–English scores (~0.85).

## Part B: Implications for Bilingual NLP in the MENA Region

These results have direct implications for deploying NLP systems in the Middle East and North
Africa. The fact that topically related English–Arabic pairs score measurably higher than
unrelated pairs (e.g., 0.75 vs. 0.48) means that a single `bert-base-multilingual-cased`
model can realistically power bilingual semantic search or document retrieval — an Arabic
user querying about "اتفاق المناخ الدولي" would surface relevant English-language IPCC
documents above unrelated English texts. This removes the need to maintain separate
monolingual pipelines for Arabic and English, which significantly reduces infrastructure
cost. However, the ~10-point gap between cross-lingual and within-language similarity scores
suggests that precision will be lower than a dedicated Arabic model. For high-stakes
applications such as climate policy retrieval or legal document matching, it would be worth
fine-tuning on top of the multilingual backbone using an Arabic-focused model like
`aubmindlab/bert-base-arabertv02` to close this gap while retaining cross-lingual capability.
For lower-stakes use cases — bilingual news search, topic clustering, or content
recommendation — the multilingual model alone is likely sufficient for production deployment.