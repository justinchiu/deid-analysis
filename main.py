import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import Counter
import spacy

sns.set_theme(style="white", font_scale=1.5)

df = pd.read_parquet("wikibio_redacted_3.parquet.gzip")


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
neural = "ts_0.10__mpw0.40__min_idf1.00"

# Analysis 1
# Reidentification model diversity. How diverse is the reid ensemble?
# * Data
#     * for each doc, top-k list of profiles for each model.
#     * ideally with p(y | x,z) for each model?
# * Metrics
#     * average size of pairwise intersection between top-k
#     * variance of the rank of hat{y} over models, averaged over docs
#     * average pairwise absolute difference in rank of hat{y} over models, averaged over docs

# TOP-K INTERSECTION ANALYSIS

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

    N = len(df)
    print(f"Intersection of top-{k} profiles")
    print()
    print("deid method")
    print("model pair: sum of intersections / num documents (avg intersection size)")
    print()
    for method, counter in intersections.items():
        print(method)
        total = (df.deid_method == method).sum()
        for pair, count in counter.items():
            print(f"{pair}: {count} / {total} ({count / total:.2f})")
        print()

#topk_analysis(10)
#topk_analysis(1)


# RANK ANALYSIS
def pairwise_diff(xs):
    n = len(xs)
    return [
        abs(xs[i] - xs[j])
        for i in range(n) for j in range(i+1,n)
    ]

def rank_analysis():
    rank_diffs = {method: Counter() for method in deid_methods}
    for row in df.iloc:
        ptrank = row["pmlm_tapas__true_profile_idxs"]
        rtrank = row["roberta_tapas__true_profile_idxs"]
        rrrank = row["roberta_roberta__true_profile_idxs"]

        deid_method = row["deid_method"]

        pt_rt, pt_rr, rt_rr = pairwise_diff([ptrank, rtrank, rrrank])
        rank_diffs[deid_method]["pt_rt"] += pt_rt
        rank_diffs[deid_method]["pt_rr"] += pt_rr
        rank_diffs[deid_method]["rt_rr"] += rt_rr

    N = len(df)
    print(f"Rank differences of true profile")
    print()
    print("deid method")
    print("model pair: avg rank diff")
    print()
    for method, counter in rank_diffs.items():
        print(method)
        total = (df.deid_method == method).sum()
        for pair, count in counter.items():
            print(f"{pair}: {count / total:.2f}")
        print()

rank_analysis()

def rank_diff_analysis():
    model_pairs = [
        (models[i], models[j])
        for i in range(len(models)) for j in range(i+1,len(models))
    ]
    print("RANK DIFFERENCES")
    for m1, m2 in model_pairs:
        print(m1, m2)
        m1_profile = f"{m1}__true_profile_idxs"
        m2_profile = f"{m2}__true_profile_idxs"

        r1 = df[m1_profile]
        r2 = df[m2_profile]
        mean_rank_diff = (r1 - r2).abs().mean()

        # only when method is correct
        #r1c = df[][m1_profile]
        #r2c = df[][m2_profile]
        #mean_rank_diff_filtered = (r1c - r2c).abs().mean()


def correlation_analysis():
    print("CORRELATION OF TRUE PROFILE RANK")
    for m1, m2 in model_pairs:
        print(m1, m2)
        m1_profile = f"{m1}__true_profile_idxs"
        m2_profile = f"{m2}__true_profile_idxs"
        corr = df.groupby("deid_method")[[m1_profile, m2_profile]].corr()
        #log_corr = np.log(df.groupby("deid_method")[[m1_profile, m2_profile]]).corr()
        for method in deid_methods:
            correlation = corr.loc[method].values[0,1]
            print(f"  {method}: corr {correlation}")
            #log_correlation = log_corr.loc[method].values[0,1]
            #print(f"  {method}: corr {correlation} || corr(log) {log_correlation}")

def plot_ranks(clipval=10000):
    df["pmlm_tapas__true_profile_idxs__clipped"] = df["pmlm_tapas__true_profile_idxs"].clip(0,clipval)
    df["roberta_tapas__true_profile_idxs__clipped"] = df["roberta_tapas__true_profile_idxs"].clip(0,clipval)
    df["roberta_roberta__true_profile_idxs__clipped"] = df["roberta_roberta__true_profile_idxs"].clip(0,clipval)

    df["pmlm_tapas__true_profile_idxs__log"] = np.log10(1 + df["pmlm_tapas__true_profile_idxs"].clip(0,clipval))
    df["roberta_tapas__true_profile_idxs__log"] = np.log10(1+df["roberta_tapas__true_profile_idxs"].clip(0,clipval))
    df["roberta_roberta__true_profile_idxs__log"] = np.log10(1+df["roberta_roberta__true_profile_idxs"].clip(0,clipval))

    g = sns.jointplot(
        data=df[df["deid_method"] == neural],
        x="pmlm_tapas__true_profile_idxs__clipped",
        #x="roberta_roberta__true_profile_idxs__clipped",
        y="roberta_tapas__true_profile_idxs__clipped",
        s=25, marginal_kws=dict(bins=25, fill=True),
    )
    #plt.tight_layout()
    plt.savefig(f"rank-rank-{clipval}.png")
    #plt.tight_layout()
    g = sns.jointplot(
        data=df[df["deid_method"] == neural],
        x="pmlm_tapas__true_profile_idxs__log",
        #x="roberta_roberta__true_profile_idxs__clipped",
        y="roberta_tapas__true_profile_idxs__log",
        s=25, marginal_kws=dict(bins=25, fill=True),
    )
    g.ax_joint.set_ylabel("ReID Model (RT)")
    g.ax_joint.set_xlabel("DeID (PT @ K = 8)")

    #g.ax_joint.set_xlim(1.5, math.log10(1+clipval) + .2)
    g.ax_joint.set_xlim(1.5)
    #g.ax_joint.set_xlim(1, )
    #g.ax_joint.set_ylim(-.1, math.log10(1+clipval) + .1)

    #g.ax_joint.set_xticks([1,2,3])
    g.ax_joint.set_xticks([2,3])
    g.ax_joint.set_yticks([1,2,3])

    #g.ax_joint.set_xticklabels([10,100,1000])
    g.ax_joint.set_xticklabels([100,1000])
    g.ax_joint.set_yticklabels([10,100,1000])

    plt.savefig(f"logrank-logrank-{clipval}.png")

for clip in [100, 1000, 2000, 10000]:
    plot_ranks(clip)

# Analysis 2
# De-identification strategy. How does greedy deid redaction differ from baselines?
# * Data
#     * masked documents under each method
# * Metrics
#     * tbd after exploratory analysis?
#     * difference in named entities redacted
#     * quantify quasi-identifiers?

# 0-999, eg range(1000)
documents = df["i"].unique()
ndocs = 1000

for doc in range(ndocs):
    rows = df[df["i"] == doc]
    texts = rows["perturbed_text"]

    text_row = rows[rows["deid_method"] == "document"]
    idf_row = rows[rows["deid_method"] == "idf__maxidf_7.0"]
    nn_row = rows[rows["deid_method"] == "ts_0.10__mpw0.40__min_idf1.00"]

    text = text_row["perturbed_text"]
    idf_text = idf_row["perturbed_text"]
    nn_text = nn_row["perturbed_text"]
    #import pdb; pdb.set_trace()
