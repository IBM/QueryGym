import psycopg2
import re
from pandas import read_sql_query, read_sql_table
import traceback
import sys
import sqlite3

def get_connection(path):
    conn = sqlite3.connect(path, uri=path.startswith("file:"), check_same_thread=False)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    return conn

def get_db_schema(conn):
    cur = conn.cursor()
    # Get all table names (ignore sqlite internal)
    cur.execute("""
        SELECT name 
        FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
    """)
    tables = [r[0] for r in cur.fetchall()]

    table_name_to_idx = {t: i for i, t in enumerate(tables)}

    column_names_original = [[-1, "*"]]
    column_names = [[-1, "*"]]
    column_types = ["text"]

    # Collect all columns
    for t in tables:
        cur.execute(f"PRAGMA table_info('{t}')")
        cols = cur.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
        for c in cols:
            column_names_original.append([table_name_to_idx[t], c[1]])
            column_names.append([table_name_to_idx[t], c[1]])
            column_types.append(c[2].lower() if c[2] else "text")

    # Primary keys
    pk_indices = []
    for t in tables:
        cur.execute(f"PRAGMA table_info('{t}')")
        cols = cur.fetchall()
        for c in cols:
            if c[5] == 1:  # pk flag
                idx = next(
                    i for i, x in enumerate(column_names_original)
                    if x[0] == table_name_to_idx[t] and x[1] == c[1]
                )
                pk_indices.append(idx)

    # Foreign keys
    fks = []
    for t in tables:
        cur.execute(f"PRAGMA foreign_key_list('{t}')")
        fk_rows = cur.fetchall()
        for fk in fk_rows:
            (_, _, ref_table, from_col, to_col, *_ ) = fk
            try:
                src_idx = next(
                    i for i, x in enumerate(column_names_original)
                    if x[0] == table_name_to_idx[t] and x[1] == from_col
                )
                tgt_idx = next(
                    i for i, x in enumerate(column_names_original)
                    if x[0] == table_name_to_idx[ref_table] and x[1] == to_col
                )
                fks.append([src_idx, tgt_idx])
            except StopIteration:
                continue

    cur.close()
    # conn.close()

    return {
        "table_names_original": tables,
        "table_names": tables,  # can prettify later
        "column_names_original": column_names_original,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": pk_indices,
        "foreign_keys": fks
    }


def execute_sql_query(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    res = cursor.fetchall()
    description = cursor.description
    cursor.close()
    # conn.close()
    return res, description

