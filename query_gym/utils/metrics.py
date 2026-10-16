"""CODE FROM https://github.com/IBM/unitxt/blob/text2sql-metrics-update/src/unitxt/text2sql_utils.py"""

import functools
import glob
import hashlib
import json
import os
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.exceptions import ConnectionError, ReadTimeout


def compare_dfs_ignore_colnames_ordered_rows(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> bool:
    df1.fillna(0, inplace=True)
    df2.fillna(0, inplace=True)
    if df1.shape != df2.shape:
        return False
    df1_sorted_rows = np.array([np.sort(row) for row in df1.values.astype(str)])
    df2_sorted_rows = np.array([np.sort(row) for row in df2.values.astype(str)])
    return np.array_equal(df1_sorted_rows, df2_sorted_rows)


def compare_dfs_ignore_colnames_unordered_rows(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> bool:
    if df1.shape != df2.shape:
        return False
    df1_sorted = np.sort(np.sort(df1.values.astype(str), axis=1), axis=0)
    df2_sorted = np.sort(np.sort(df2.values.astype(str), axis=1), axis=0)
    return np.array_equal(df1_sorted, df2_sorted)

def compare_dfs_ignore_colnames_subset_slow(
    df1: pd.DataFrame, df2: pd.DataFrame, ignore_row_order: bool = True
) -> bool:
    def row_to_multiset(row):
        return Counter(str(x) for x in row)

    if df1.empty or df2.empty or len(df1) != len(df2):
        return False

    subset_df, superset_df = (df1, df2) if df1.shape[1] <= df2.shape[1] else (df2, df1)
    subset_rows = [row_to_multiset(row) for row in subset_df.values]
    superset_rows = [row_to_multiset(row) for row in superset_df.values]

    if ignore_row_order:
        used = [False] * len(superset_rows)
        for r1 in subset_rows:
            found = False
            for i, r2 in enumerate(superset_rows):
                if not used[i] and all(r1[k] <= r2.get(k, 0) for k in r1):
                    used[i] = True
                    found = True
                    break
            if not found:
                return False
        return True

    for r1, r2 in zip(subset_rows, superset_rows):
        if not all(r1[k] <= r2.get(k, 0) for k in r1):
            return False
    return True


def compare_dfs_ignore_colnames_subset3(
    df1: pd.DataFrame, df2: pd.DataFrame, ignore_row_order: bool = True
) -> bool:
    def row_to_multiset(row):
        return Counter(str(x) for x in row)

    def rows_to_multisets(df):
        return [row_to_multiset(row) for row in df.values]

    def sort_df(df):
        sorted_df = df.copy()
        for col in sorted_df.columns:
            sorted_df[col] = sorted_df[col].astype(str).sort_values(ignore_index=True)
        return sorted_df

    if df1.empty or df2.empty or len(df1) != len(df2):
        return False

    subset_df, superset_df = (df1, df2) if df1.shape[1] <= df2.shape[1] else (df2, df1)

    if ignore_row_order:
        subset_df = sort_df(subset_df)
        superset_df = sort_df(superset_df)

    subset_rows = rows_to_multisets(subset_df)
    superset_rows = rows_to_multisets(superset_df)

    for r1, r2 in zip(subset_rows, superset_rows):
        if not all(r1[k] <= r2.get(k, 0) for k in r1):
            return False
    return True


def compare_dfs_ignore_colnames_subset2(
    df1: pd.DataFrame, df2: pd.DataFrame, ignore_row_order: bool = True
) -> bool:
    def row_to_multiset(row):
        return Counter(str(x) for x in row)

    def rows_to_multisets(df):
        return [row_to_multiset(row) for row in df.values]
    
    def sort_df(df):
        return df.sort_values(by=list(df.columns)).reset_index(drop=True)

    if df1.empty or df2.empty or len(df1) != len(df2):
        return False

    subset_df, superset_df = (df1, df2) if df1.shape[1] <= df2.shape[1] else (df2, df1)

    if ignore_row_order:
        subset_df = sort_df(subset_df)
        superset_df = sort_df(superset_df)

    subset_rows = rows_to_multisets(subset_df)
    superset_rows = rows_to_multisets(superset_df)

    for r1, r2 in zip(subset_rows, superset_rows):
        if not all(r1[k] <= r2.get(k, 0) for k in r1):
            return False
    return True

def compare_dfs_ignore_colnames_subset(
    df1: pd.DataFrame, df2: pd.DataFrame, ignore_row_order: bool = True
) -> bool:
    def row_to_multiset(row):
        return Counter(str(x) for x in row)

    def normalize_and_sort(df):
        return pd.DataFrame(
            sorted([sorted(map(str, row)) for row in df.values.tolist()])
        )

    if df1.empty or df2.empty or len(df1) != len(df2):
        return False

    subset_df, superset_df = (df1, df2) if df1.shape[1] <= df2.shape[1] else (df2, df1)

    if ignore_row_order:
        subset_df = normalize_and_sort(subset_df)
        superset_df = normalize_and_sort(superset_df)

    subset_rows = [row_to_multiset(row) for row in subset_df.values]
    superset_rows = [row_to_multiset(row) for row in superset_df.values]

    for r1, r2 in zip(subset_rows, superset_rows):
        if not all(r1[k] <= r2.get(k, 0) for k in r1):
            return False
    return True


def compare_result_dfs(
    gold_df: pd.DataFrame, pred_df: pd.DataFrame, gold_sql = None
) -> Tuple[int, int, int]:

    row_correctness = 0
    subset_match = 0
    non_empty_match = 0
    if len(gold_df) == len(pred_df):
        row_correctness=1
    if isinstance(gold_sql, str) and "ORDER BY" in gold_sql.upper():
        match = int(compare_dfs_ignore_colnames_ordered_rows(pred_df, gold_df))
        if not match and not gold_df.empty and not pred_df.empty:
            non_empty_match = match
            if compare_dfs_ignore_colnames_subset3(
                gold_df, pred_df, ignore_row_order=False
            ):
                subset_match = 1
    else:
        match = int(compare_dfs_ignore_colnames_unordered_rows(pred_df, gold_df))
        if not match and not gold_df.empty and not pred_df.empty:
            non_empty_match = match
            if compare_dfs_ignore_colnames_subset3(
                gold_df, pred_df, ignore_row_order=True
            ):
                subset_match = 1
    return match, non_empty_match, subset_match, row_correctness
