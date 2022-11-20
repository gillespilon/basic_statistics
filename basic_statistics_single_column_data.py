#! /usr/bin/env python3
"""
Examples of statistics from a single column of data
"""

import scipy.stats as stats
import datasense as ds
import pandas as pd
import numpy as np
import math


def main():
    data = {
        "y": [
            61.3, 71.4, 75.4, 54.6, 65, 88.8, 46.6, 91.4, 68, 61.1,
            62.1, 98.6, 70.6, 76.5, 72.1, 61.6, 79.8, 78.1, 66.5, 66.9,
            75.2, 83.4, 80.3, 64.1, 72.6, 82.8, 83, 74.1, 53.6, 56.1,
            73.3, 62.4, 69.7, 57.4, 58.5, 73.6, 63.1, 63.7, 68.6, 91.2,
            87.7, 67.1, 81.8, 79.7, 82.1, 74.7, 91, 79.7, 48.9, 71.2
        ]
    }
    header_title = "Basic statistics for a single column of data"
    output_url = "basic_statistics_single_column_data.html"
    header_id = "basic-statistics-single-column-data"
    original_stdout = ds.html_begin(
        output_url=output_url,
        header_title=header_title,
        header_id=header_id
    )
    series_name_y = "y"
    df = pd.DataFrame(data=data)
    print("pandas.Series.describe():")
    print()
    print(df[series_name_y].describe())
    print()
    print("Number of values         :", df[series_name_y].count())
    mean = df[series_name_y].mean()
    print("Average                  :", mean)
    print("Sample standard deviation:", df[series_name_y].std())
    print("Sample variance          :", df[series_name_y].var())
    print("Minimum value            :", df[series_name_y].min())
    print("Maximum value            :", df[series_name_y].max())
    print()
    quantiles = np.quantile(
        a=df[series_name_y],
        q=[.01, .05, .1, .25, .5, .75, .9, .95, .99],
        method="median_unbiased"
    )
    print("NumPy.quantile():")
    print(
        "\n",
        "0.25", quantiles[3], "\n",
        "0.50", quantiles[4], "\n",
        "0.75", quantiles[5], "\n",
    )
    print()
    median = np.quantile(
        a=df[series_name_y],
        q=0.5,
        method="median_unbiased"
    )
    print("Median                   :", median)
    iqr = (
        np.quantile(
            a=df[series_name_y],
            q=.75,
            method="median_unbiased"
        ) -
        np.quantile(
            a=df[series_name_y],
            q=.25,
            method="median_unbiased"
        )
    )
    print("Interquartile range      :", iqr)
    print(
        "95 % CI of the median    :",
        median - 1.57 * iqr / math.sqrt(df[series_name_y].count()),
        median + 1.57 * iqr / math.sqrt(df[series_name_y].count()),
    )
    print()
    print("Average                  :", mean)
    ciaverage = stats.t.interval(
            confidence=0.95,
            df=len(df[series_name_y])-1,
            loc=mean,
            scale=stats.sem(df[series_name_y])
        )
    print("95 % CI of average       :", ciaverage[0], ciaverage[1])
    print()
    print()
    print()
    print("Statistics using datasense")
    print()
    series = ds.parametric_summary(series=df[series_name_y])
    print("ds.parametric_summary()")
    print(series)
    print()
    print("ds.nonparametric_summary()")
    series = ds.nonparametric_summary(series=df[series_name_y])
    print(series)
    print()
    print("numpy.quantile 0.25, 0.75 with different methods")
    print(
        "method = linear",
        np.quantile(
            a=df[series_name_y],
            q=[.25, .75],
            method="linear"
        )
    )
    print(
        "method = median_unbiased",
        np.quantile(
            a=df[series_name_y],
            q=[.25, .75],
            method="median_unbiased"
        )
    )
    ds.html_end(
        original_stdout=original_stdout,
        output_url=output_url
    )


if __name__ == "__main__":
    main()
