import re
import pandas as pd
import duckdb
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import numpy as np
import matplotlib.pyplot as plt

def clean_title(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def compute_title_ngram_trends(
    df: pd.DataFrame,
    ngram_range=(2, 2),
    min_df=100,
    max_features=1000,
    sample_size=200000,
    smooth_window=3,
    year_min=1950,
    extra_stopwords = set(),
):
    base = df.loc[
        df["year"].notna()
        & df["title"].notna()
        & (df["title"].astype(str).str.strip() != "")
        & (df["year"] >= year_min),
        ["year", "title"]
    ].copy()

    base["year"] = base["year"].astype(int)

    stopwords = sorted(set(ENGLISH_STOP_WORDS) | set(extra_stopwords))

    sample_n = min(sample_size, len(base))
    sample = base.sample(sample_n, random_state=42) if sample_n < len(base) else base

    selector = CountVectorizer(
        stop_words=stopwords,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
    )
    selector.fit(sample["title"])
    terms = selector.get_feature_names_out()

    counter = CountVectorizer(
        stop_words=stopwords,
        ngram_range=ngram_range,
        vocabulary=terms,
    )

    yearly_totals = base.groupby("year").size().sort_index()
    yearly_rows = []

    for year, group in base.groupby("year", sort=True):
        X_year = counter.transform(group["title"])
        counts = np.asarray(X_year.sum(axis=0)).ravel()
        yearly_rows.append(pd.Series(counts, index=terms, name=year))

    yearly_counts = pd.DataFrame(yearly_rows).fillna(0).sort_index()
    yearly_share = yearly_counts.div(yearly_totals, axis=0).fillna(0)

    if smooth_window and smooth_window > 1:
        yearly_share = yearly_share.rolling(
            window=smooth_window,
            center=True,
            min_periods=1
        ).mean()

    totals = yearly_counts.sum(axis=0).sort_values(ascending=False)


    return yearly_share, totals


def plot_phrase_trends(yearly_share, terms, title, ylabel="Share of titles"):
    plt.figure(figsize=(14, 7))

    for term in terms:
        if term in yearly_share.columns:
            plt.plot(yearly_share.index, yearly_share[term] * 100, label=term)

    plt.xlabel("Year")
    plt.ylabel(f"{ylabel} (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Phrase", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_top(yearly_share, totals, top_n=10):
    top_overall = totals.head(top_n).index.tolist()

    plot_phrase_trends(
        yearly_share,
        top_overall,
        title=f"Top {top_n} most common title phrases over time",
    )