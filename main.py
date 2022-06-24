import pandas as pd

pt = "pmlm_tapas"
rt = "roberta_tapas"
rr = "roberta_roberta"

models = [pt, rt, rr]

idxs = [f"{model}__topk_idxs" for model in models]
values = [f"{model}__topk_values" for model in models]
correct = [f"{model}__was_correct" for model in models]

df = pd.read_parquet("wikibio_redacted_2.parquet.gzip")
import pdb; pdb.set_trace()

# Analysis 1
# Reidentification model diversity. How diverse is the reid ensemble?
# * Data
#     * for each doc, top-k list of profiles for each model.
#     * ideally with p(y | x,z) for each model?
# * Metrics
#     * variance of the rank of hat{y} over models, averaged over docs
#     * average pairwise absolute difference in rank of hat{y} over models, averaged over docs




# Analysis 2
# De-identification strategy. How does greedy deid redaction differ from baselines?
# * Data
#     * masked documents under each method
# * Metrics
#     * tbd after exploratory analysis?
#     * difference in named entities redacted
#     * quantify quasi-identifiers?
