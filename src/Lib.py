import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def extract_from_data(df: pd.DataFrame, region: str, parameters: list) -> pd.DataFrame:
    mask = (
        (df['region'] == region) &
        (df['parameter'].isin(parameters))
    )
    sub = df.loc[mask, ['year', 'parameter', 'value']]
    # sum duplicates
    summed = sub.groupby(['year', 'parameter'], as_index=False)['value'].sum()
    # pivot wide
    pivot = summed.pivot(index='year', columns='parameter', values='value').fillna(0)
    # reset index
    result = pivot.reset_index()[['year'] + parameters].sort_values('year')
    return result


def compare(df: pd.DataFrame, region: str, p1: str, p2: str):
    data = extract_from_data(df, region, [p1, p2])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(data['year'], data[p1], marker='o', linewidth=2, label=p1)
    ax1.set_xlabel('Year')
    ax1.set_ylabel(p1)

    ax2 = ax1.twinx()
    ax2.plot(data['year'], data[p2], marker='s', linewidth=2, label=p2)
    ax2.set_ylabel(p2)

    # combine legends
    lns1, labs1 = ax1.get_legend_handles_labels()
    lns2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1 + lns2, labs1 + labs2, loc='upper left')

    plt.title(f'{p1} and {p2} Over Time in {region}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def compare_linked(df: pd.DataFrame, region: str, p1: str, p2: str):
    data = extract_from_data(df, region, [p1, p2])
    # compute % of initial
    pct = data.copy()
    pct[p1] = (pct[p1] / pct[p1].iloc[0]) * 100
    pct[p2] = (pct[p2] / pct[p2].iloc[0]) * 100

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(pct['year'], pct[p1], marker='o', linewidth=2, label=p1)
    ax1.set_xlabel('Year')
    ax1.set_ylabel(f'{p1} (% of initial)')
    ax1.set_ylim(100, pct[[p1,p2]].values.max() * 1.1)

    ax2 = ax1.twinx()
    ax2.plot(pct['year'], pct[p2], marker='s', linewidth=2, label=p2)
    ax2.set_ylabel(f'{p2} (% of initial)')
    ax2.set_ylim(100, pct[[p1,p2]].values.max() * 1.1)

    # combine legends
    lns1, labs1 = ax1.get_legend_handles_labels()
    lns2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1 + lns2, labs1 + labs2, loc='upper left')

    plt.title(f'{p1} and {p2} as % of Initial Value in {region}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt



def plot_param_vs_param(df, region, p1, p2, annotate_years=False):
    data = extract_from_data(df, region, [p1, p2])

    x = data[p1]
    y = data[p2]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, marker='o')
    plt.xlabel(p1)
    plt.ylabel(p2)
    plt.title(f'{p2} vs {p1} in {region}')
    plt.grid(True, linestyle='--', alpha=0.5)

    if annotate_years:
        for xi, yi, year in zip(x, y, data['year']):
            plt.text(xi, yi, str(year), fontsize=8, ha='right', va='bottom')

    plt.tight_layout()
    plt.show()
