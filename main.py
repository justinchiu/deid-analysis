import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import Counter
import spacy

import pickle
import pathlib

import re
import spacy
from spacy.tokenizer import Tokenizer

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

# spacy hacks
nlp = spacy.load('en_core_web_sm')
#text = "This is it's"
#print("Before:", [tok for tok in nlp(text)])
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

# ner model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
bert_ner_pipeline = pipeline(
  "ner", model=model, tokenizer=tokenizer, device=(0 if torch.cuda.is_available() else -1)
)

# seaborn settings
sns.set(rc = {'figure.figsize':(15,8)})
sns.set_theme(style="white", font_scale=1.5)

# data setup
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
idf = "idf__maxidf_7.0"

# Analysis 1
# Reidentification model diversity. How diverse is the reid ensemble?
# * Data
#     * for each doc, top-k list of profiles for each model.
#     * ideally with p(y | x,z) for each model?
# * Plot
#     * Rank of DeID vs rank of ReID

def plot_ranks(clipval=2000):
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

    plt.close("all")
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

    g.ax_joint.set_xlim(1.5)

    g.ax_joint.set_xticks([2,3])
    g.ax_joint.set_yticks([1,2,3])

    g.ax_joint.set_xticklabels([100,1000])
    g.ax_joint.set_yticklabels([10,100,1000])

    plt.savefig(f"logrank-logrank-{clipval}.png")
    plt.close("all")

for clip in [100, 1000, 2000, 10000]:
    plot_ranks(clip)

# Analysis 2
# De-identification strategy. How does greedy deid redaction differ from baselines?
# * Data
#     * masked documents under each method
# * Plots
#     * Histogram of named entitites
#     * Histogram of PoS

# 0-999, eg range(1000)
documents = df["i"].unique()
ndocs = 1000

n = 0

pospath = pathlib.Path("posdict.pkl")
nerpath = pathlib.Path("nerdict.pkl")

exclude_pos = ["X", "PUNCT", "DET", "ADP"]

posdict = None
nerdict = None
if not pospath.exists():
    posdict = {m: Counter() for m in deid_methods}
    nerdict = {m: Counter() for m in deid_methods}
    mask_sums = Counter()
    for doc in range(ndocs):
        rows = df[df["i"] == doc]

        if rows.shape[0] != 5:
            # missing rows
            continue

        n += 1
        #texts = rows["perturbed_text"]

        text_row = rows[rows["deid_method"] == "document"]
        lex_row = rows[rows["deid_method"] == "lexical"]
        ner_row = rows[rows["deid_method"] == "named_entity"]
        idf_row = rows[rows["deid_method"] == "idf__maxidf_7.0"]
        nn_row = rows[rows["deid_method"] == "ts_0.10__mpw0.40__min_idf1.00"]

        text = text_row["perturbed_text"].values[0]
        lex_text = lex_row["perturbed_text"].values[0]
        ner_text = ner_row["perturbed_text"].values[0]
        idf_text = idf_row["perturbed_text"].values[0]
        nn_text = nn_row["perturbed_text"].values[0]

        doc = nlp(text)
        assert len(doc) == len(text.split())
        entities = bert_ner_pipeline(text)

        #for i in range(len(doc)):
        #import pdb; pdb.set_trace()
        for method in deid_methods:
            words = rows[rows["deid_method"] == method]["perturbed_text"].values[0].split()
            is_masks = [w == "<mask>" for w in words]
            mask_sums[method] += np.sum(is_masks)
            assert len(doc) == len(is_masks)
            for i, (token, is_mask) in enumerate(zip(doc, is_masks)):
                if not is_mask:
                    pos = token.pos_
                    if pos not in exclude_pos:
                        posdict[method][pos] += 1
                    iob = token.ent_iob_
                    if iob == "B" or iob == "I":
                        label = token.ent_type_
                        nerdict[method][label] += 1
    with pospath.open("wb") as f:
        pickle.dump(posdict, f)
    with nerpath.open("wb") as f:
        pickle.dump(nerdict, f)
    print(mask_sums)
else:
    with pospath.open("rb") as f:
        posdict = pickle.load(f)
    with nerpath.open("rb") as f:
        nerdict = pickle.load(f)


pos_global = sum(posdict.values(), Counter())
ner_global = sum(nerdict.values(), Counter())

def plot_words(k=10):
    pos_list = [p for p,c in pos_global.most_common(7)]
    ner_list = [n for n,c in ner_global.most_common(7)]
    pos_method_list = [neural, idf]
    ner_method_list = [neural, idf, "lexical"]


    # count, method, counter
    pos_df = pd.DataFrame([
        (method, x, c / posdict["document"][x])
        for method, counter in posdict.items()
        for x, c in counter.items()
        if x in pos_list and method in pos_method_list
    ], columns=["method", "pos", "percent"])
    ner_df = pd.DataFrame([
        (method, x, c / nerdict["document"][x])
        for method, counter in nerdict.items()
        for x, c in counter.items()
        if x in ner_list and method in ner_method_list
    ], columns=["method", "type", "percent"])

    ax = sns.barplot(
        data=pos_df, x="pos", y="percent", hue="method",
        palette = dict(
            zip(deid_methods, sns.color_palette("hls", len(deid_methods)))
        ),
        #order = method_list,
    )
    ax.set(ylabel = "Percentage masked", xlabel = "POS tags")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        title="DeID model",
        #labels = ["NN ReID","IDF"], # ideal ordering
        labels = ["IDF", "NN ReID"], # current ordering
    )
    plt.savefig(f"pos.png")

    plt.close("all")

    ax = sns.barplot(
        data=ner_df, x="type", y="percent", hue="method",
        palette = dict(
            zip(deid_methods, sns.color_palette("hls", len(deid_methods)))
        ),
        #order = method_list,
    )
    ax.set(ylabel = "Percentage masked", xlabel = "NER labels")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        title="DeID model",
        #labels = ["NN ReID","IDF", "Lexical"], # ideal ordering
        labels = ["IDF", "NN ReID","Lexical"], # current ordering
    )
    plt.savefig(f"ner.png")
    plt.close("all")

plot_words()

print(f"Num docs analyzed: {n}")
