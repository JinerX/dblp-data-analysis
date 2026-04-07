import re
import pandas as pd
import duckdb
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Iterable, List


def clean_title(text: str) -> str:
    """
    Normalize a paper title for text processing.

    This function:
    - lowercases text
    - removes non-alphanumeric characters
    - collapses multiple spaces

    Parameters
    ----------
    text : str
        Raw title string.

    Returns
    -------
    str
        Cleaned title. Returns empty string if input is null.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def compute_title_ngram_trends(
    df: pd.DataFrame,
    ngram_range: Tuple[int,int] = (2, 2),
    min_df: int = 100,
    max_features: int =1000,
    sample_size: int=200000,
    smooth_window: int=3,
    year_min: int=1950,
    extra_stopwords: Iterable[str] = set(),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute time-based trends of n-gram usage in paper titles.

    The method:
    1. Filters valid titles and years
    2. Samples data to select frequent n-grams
    3. Builds a fixed vocabulary
    4. Counts occurrences per year
    5. Normalizes counts into yearly shares
    6. Applies optional smoothing

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with at least columns ['year', 'title'].
    ngram_range : Tuple[int, int], default=(2, 2)
        N-gram range for analysis.
    min_df : int, default=100
        Minimum document frequency threshold.
    max_features : int, default=1000
        Maximum number of features (n-grams).
    sample_size : int, default=200000
        Number of rows used for vocabulary selection.
    smooth_window : int, default=3
        Rolling window size for smoothing time series.
    year_min : int, default=1950
        Minimum year to include in analysis.
    extra_stopwords : Iterable[str], optional
        Additional stopwords to exclude.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        yearly_share : pd.DataFrame
            Rows = years, columns = n-grams, values = normalized share.
        totals : pd.Series
            Total frequency of each n-gram across all years.
    """
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


def plot_phrase_trends(yearly_share: pd.DataFrame,
                       terms: Iterable[str],
                       title: str,
                       ylabel: str="Share of titles") -> None:
    """
    Plot time trends for selected n-grams.

    Parameters
    ----------
    yearly_share : pd.DataFrame
        DataFrame with years as index and n-grams as columns.
    terms : Iterable[str]
        N-grams to plot.
    title : str
        Plot title.
    ylabel : str, default="Share of titles"
        Y-axis label.

    Returns
    -------
    None
    """
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


def plot_top(yearly_share: pd.DataFrame,
             totals: pd.Series,
             top_n: int=10) -> None:
    """
    Plot the most frequent n-grams over time.

    Parameters
    ----------
    yearly_share : pd.DataFrame
        Yearly normalized n-gram frequencies.
    totals : pd.Series
        Total n-gram counts across dataset.
    top_n : int, default=10
        Number of top phrases to plot.

    Returns
    -------
    None
    """
    top_overall = totals.head(top_n).index.tolist()

    plot_phrase_trends(
        yearly_share,
        top_overall,
        title=f"Top {top_n} most common title phrases over time",
    )