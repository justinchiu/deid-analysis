import numpy as np
import pandas as pd

from collections import Counter


df = pd.read_parquet("wikibio_redacted_2.parquet.gzip")


pt = "pmlm_tapas"
rt = "roberta_tapas"
rr = "roberta_roberta"

models = [pt, rt, rr]

idxs = [f"{model}__topk_idxs" for model in models]
values = [f"{model}__topk_values" for model in models]
correct = [f"{model}__was_correct" for model in models]

pt_idxs, rt_idxs, rr_idxs = idxs
pt_values, rt_values, rr_values = values
pt_correct, rt_correct, rr_correct = correct


deid_methods = df["deid_method"].unique()


# Analysis 1
# Reidentification model diversity. How diverse is the reid ensemble?
# * Data
#     * for each doc, top-k list of profiles for each model.
#     * ideally with p(y | x,z) for each model?
# * Metrics
#     * variance of the rank of hat{y} over models, averaged over docs
#     * average pairwise absolute difference in rank of hat{y} over models, averaged over docs

def pairwise_intersection(xs):
    n = len(xs)
    intersection_lens = [
        len(xs[i].intersection(xs[j]))
        for i in range(n) for j in range(i+1,n)
    ]
    return intersection_lens

#method = "pmlm_tapas__eps_0.0005"
#rows = df[df["deid_method"] == method]
#row = rows.iloc[0]

def topk_analysis(k=10):
    intersections = {method: Counter() for method in deid_methods}
    for row in df.iloc:
        ptids = row[pt_idxs][:k]
        rtids = row[rt_idxs][:k]
        rrids = row[rr_idxs][:k]

        deid_method = row["deid_method"]

        pt_rt, pt_rr, rt_rr = pairwise_intersection([set(x) for x in [ptids, rtids, rrids]])
        intersections[deid_method]["pt_rt"] += pt_rt
        intersections[deid_method]["pt_rr"] += pt_rr
        intersections[deid_method]["rt_rr"] += rt_rr
        if deid_method == "document":
            import pdb; pdb.set_trace()

    N = len(df)
    print(f"Num examples: {N}")
    print("Intersection of top-{k} profiles")
    print()
    print("method")
    print("pair: total (avg)")
    for method, counter in intersections.items():
        print()
        print(method)
        for pair, count in counter.items():
            print(f"{pair}: {count} ({count / N:.2f})")

topk_analysis(10)
topk_analysis(1)
import pdb; pdb.set_trace()



# Analysis 2
# De-identification strategy. How does greedy deid redaction differ from baselines?
# * Data
#     * masked documents under each method
# * Metrics
#     * tbd after exploratory analysis?
#     * difference in named entities redacted
#     * quantify quasi-identifiers?
