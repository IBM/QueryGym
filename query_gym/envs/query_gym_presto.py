from collections import defaultdict
import os
import random
from enum import Enum, StrEnum
from typing import Iterable, TypeVar
import re
import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
import traceback
from ..utils.metrics import compare_result_dfs
from .query_gym_sqlite import SQLiteQueryGym

from pandas import read_sql_query, read_sql_table
from urllib.parse import urlparse, unquote



class PrestoQueryGym(SQLiteQueryGym):

    def get_db_connection(self, **kwargs):
        parsed = urlparse(self.dataset_item.path)

        host = parsed.hostname
        port = parsed.port
        username = unquote(parsed.username)
        password = parsed.password
        
        # First path segment = catalog, second = schema
        path_parts = [p for p in parsed.path.split("/") if p]
        catalog = path_parts[0] if len(path_parts) > 0 else None
        schema = path_parts[1] if len(path_parts) > 1 else None
        
        # Connect with prestodb
        import prestodb
        
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
        return conn
            
    def run_presto_sql_to_df(self, sql):
        rows, desc = self.execute_query(sql)
        return pd.DataFrame(rows, columns=[col[0] for col in desc])
        
        
    def get_tables(self, **kwargs):
        cur = self.conn.cursor()
        cur.execute("SHOW TABLES")
        tables = cur.fetchall()
        
        out = {tbl[0] : self.run_presto_sql_to_df(f"SELECT * from {tbl[0]}") for tbl in tables}
        # TODO: Adding this as the current presto instance has extra table.
        del out['custom_table']
        cur.close()
        return out

  
 
    def reset(  # pylint: disable=arguments-differ
        self,
        idx=None,
        seed=None,
        options=None,
        logger=None
    ) -> str:
        if idx == None:
            random.seed(seed)
            idx = random.randrange(len(self.dataset))
        self.idx = idx
        self.dataset_item = self.dataset[self.idx]
        self.data = self.dataset._data[self.idx]
        self.data['path'] =  self.dataset_item.path
        self.data['idx'] = idx
        self.data['steps'] = 0
        self.data['SQL'] = self.dataset_item.gt
        self.data['operations'] = 0
        self.data['operation_fail'] = 0
        self.data['trace'] = {}
        self.generated_sql = ""
        self.prefix_tracker = defaultdict(int)
        self.conn = self.get_db_connection()            
        self.cte =  self.get_tables()
        self.db = {'table_names_original':list(self.cte.keys())}
        self.gt = self.get_gt()
        return self._get_observation()

    def execute_query(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        res = cursor.fetchall()
        description = cursor.description
        cursor.close()
        return res, description
        
        
    def _get_schema(self):
        schema_list = [self.execute_query(f"SHOW CREATE TABLE {tbl}")[0][0][0] for tbl in self.db['table_names_original']]
        return "## Schema\n\n"+"\n\n".join(schema_list)
    
    def _get_columns(self, table_id):
        if table_id in self.db['table_names_original']:
            rows, col = self.execute_query(f"SHOW COLUMNS from {table_id}")
            columns =[ f"{n}. {r[0]} --- {r[3]}" if len(r[3]) > 0 else  f"{n}. {r[0]}" for n, r in enumerate(rows) ]
            return f"Table {table_id} has {len(columns)} columns. The column indexes, names, and descriptions are listed below.\n" + " \n".join(columns)
        elif table_id in self.cte:
            columns = self.cte[table_id].columns
            return f"Table {table_id} is a temporary result set, commonly known as Common Table Expression (CTE). The exact SQL defintion of {table_id} is:\n```\n{self.data['trace'][table_id]}\n```\n It has {len(columns)} columns. The column names are listed below.\n" + " \n".join(columns)
        else:
            return f"Table {table_id} was not found." + self._get_tables()
    
    def _get_columns_type(self, table_id):
        if table_id in self.db['table_names_original'] :
            rows, col = self.execute_query(f"SHOW COLUMNS from {table_id}")
            columns =[ f"{n}. {r[0]}: {r[1]}" for n, r in enumerate(rows) ]
            return f"Table {table_id} has {len(columns)} columns. The column indexes, names, and their data types are listed below.\n" + " \n".join(columns)
        elif table_id in self.cte.keys():
            column_type = self.cte[table_id].dtypes.to_markdown()
            return f"Table {table_id} is a temporary result set, commonly known as Common Table Expression (CTE). The exact SQL defintion of {table_id} is:\n```\n{self.data['trace'][table_id]}\n```\n It has {len(self.cte[table_id].columns)} columns. The column names and their types are listed below.\n" + column_type
        else:
            return f"Table {table_id} was not found." + self._get_tables()
        