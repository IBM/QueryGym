from query_gym.dataset.unite_dataset import UniteDataset
import unittest
import pandas as pd

from pandas import read_sql_query, read_sql_table
import os 
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

UNITE_DATASET_PATH ="/data/datasets/unified/"

datasets = {
        'wikisql':  ['test', 'train', 'dev'],
        "Criteria2SQL": ['test', 'train', 'dev'],
        "acl_sql": ['test'],
        "cosql": ['dev','train'],
        "fiben": ['dev'],
        "seoss": ['test'],
        "sparc": ["dev","train"],
        # "spider": ['dev','train_spider', 'train_others'],
        "spider_syn": ['dev','train_spider'],
        "squall/": ['dev','train'],
        }   

class SqliteConnectionTest(unittest.TestCase):
    
    def test_read_sqlite(self):
        import sqlite3
        dataset=  UniteDataset(
                dataset_path=f"{UNITE_DATASET_PATH}/sparc",
                mode="dev",
            )
        tables = dataset.tables['wta_1']['table_names_original']
        path = f"file:{UNITE_DATASET_PATH}/sparc/database/wta_1/wta_1.sqlite?mode=ro"
        conn = sqlite3.connect(path, uri=path.startswith("file:"))
        conn.text_factory = lambda b: b.decode(errors = 'ignore')
        tables_list =  list(read_sql_query("SELECT *  FROM sqlite_master WHERE type='table';", conn)['name'])
        assert tables == tables_list
        print("\n")
        create_table = list(read_sql_query("SELECT  sql FROM sqlite_master WHERE type='table';", conn)['sql'])[0]
        conn.close()
        print(create_table)
        assert "CREATE TABLE players" in create_table


class PrestoConnectionTest(unittest.TestCase):
    
    def test_read(self):
        from urllib.parse import urlparse, unquote
        import prestodb
        
        conn_url = os.environ.get("PRESTO_CONNECTION_STRING")
        # Parse the URL
        parsed = urlparse(conn_url)

        host = parsed.hostname
        port = parsed.port
        username = unquote(parsed.username)
        password = parsed.password
        # First path segment = catalog, second = schema
        path_parts = [p for p in parsed.path.split("/") if p]
        catalog = path_parts[0] if len(path_parts) > 0 else None
        schema = path_parts[1] if len(path_parts) > 1 else None

        # Connect with prestodb
        conn = prestodb.dbapi.connect(
            host=host,
            port=port,
            user=username,
            catalog=catalog,
            schema=schema,
            http_scheme="https",
            auth=prestodb.auth.BasicAuthentication(username, password),
        )
        
        # Disable TLS certificate verification (not recommended in prod)
        conn._http_session.verify = False

        cur = conn.cursor()
        cur.execute("SHOW CREATE TABLE docset3_dl3_inv_t10_entities")
        rows = cur.fetchall()
        print(rows[0][0])
        cur.execute("SHOW SCHEMAS")
        rows = cur.fetchall()
        print(rows)
        conn.close()
        


    def test_presto_sqlalchemy(self):
        from urllib.parse import urlparse, unquote,  parse_qs
        import prestodb
        
        def _parse_presto_sqlalchemy_url(conn_str: str) -> dict:
            """
            Parse a SQLAlchemy-style Presto URL into prestodb.dbapi.connect(**kwargs) args.

            Example input:
            presto://user:pass@host:30624/catalog/schema?currentSchema=schema
            """
            u = urlparse(conn_str)
            username = unquote(u.username or "")
            password = unquote(u.password or "")
            host = u.hostname
            port = u.port or 443  # default to 443 for https
            # path like "/catalog/schema"
            path_parts = [p for p in (u.path or "").split("/") if p]
            catalog = path_parts[0] if len(path_parts) >= 1 else None
            schema = path_parts[1] if len(path_parts) >= 2 else None

            qs = parse_qs(u.query or "")
            # Allow overriding schema via ?currentSchema=...
            if "currentSchema" in qs and qs["currentSchema"]:
                schema = qs["currentSchema"][0]

            # Build BasicAuthentication if password is provided
            import prestodb  # local import to keep optional

            auth = prestodb.auth.BasicAuthentication(username, password) if password else None

            # Filter out Nones
            kwargs = {
                "host": host,
                "port": port,
                "user": username,
                "catalog": catalog,
                "schema": schema,
                "http_scheme": "https",
                "auth": auth,
                "source": "text2sql-eval",
            }
            return {k: v for k, v in kwargs.items() if v is not None}

        def _run_presto_sql_to_df(sql, conn):
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall() or []
            cols = [d[0] for d in (cur.description or [])]
            cur.close()
            return pd.DataFrame(rows, columns=cols)


        
        conn_url = os.environ.get("PRESTO_CONNECTION_STRING")
        base_connect_kwargs = _parse_presto_sqlalchemy_url(conn_url)
        def _ping():
            conn = prestodb.dbapi.connect(**base_connect_kwargs)
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchall()
            cur.close()
            conn.close()
        _ping()
        print("Connection Check Successfule")
        # Parse the URL
        conn = prestodb.dbapi.connect(**base_connect_kwargs)
        
        # Disable TLS certificate verification (not recommended in prod)
        conn._http_session.verify = False

        cur = conn.cursor()
        cur.execute("SHOW CREATE TABLE docset3_dl3_inv_t10_entities")
        rows = cur.fetchall()
        print(rows[0][0])
        cur.execute("SHOW SCHEMAS")
        rows = cur.fetchall()
        print(rows)
        cur = conn.cursor()
        cur.execute("SHOW TABLES")
        tables = cur.fetchall()
        out = {tbl[0] : _run_presto_sql_to_df(f"SELECT * from {tbl[0]}", conn) for tbl in tables}
        cur.close()
        conn.close()
    
        
if __name__ == '__main__':
    unittest.main()
    