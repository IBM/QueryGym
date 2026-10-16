import traceback
import pandas as pd
from pandas import read_sql_query

from ..db_utils.sqlite_utils import (execute_sql_query, get_connection,
                                     get_db_schema)
from .query_gym_env import QueryGym


class SQLiteQueryGym(QueryGym):
    def get_db_connection(self, path=None, **kwargs):
        path = self.dataset_item.path
        if not(hasattr(self, "prev_path") and path == self.prev_path) :
            # avoid loading if its same.
            if self.conn is not None: 
                self.conn.close()
            conn = get_connection(path)
        return conn
    
    def close_db_connection(self, conn=None, **kwargs):
        if self.conn != None:
            self.conn.close()
    
    def get_tables(self, **kwargs):
        tables = self.db['table_names_original']
        out = {tbl : read_sql_query(f"SELECT * from `{tbl}`", self.conn) for tbl in tables}
        return out
 
    def get_gt(self):
        try:
            gt_data, gt_description = self.execute_query(self.dataset_item.gt)
            gt_columns = [i[0] for i in gt_description]
            return pd.DataFrame(gt_data, columns=gt_columns)
        except Exception as e:
            return traceback.format_exception_only(e)[0]
            
    def execute_query(self, query):
        return execute_sql_query(self.conn, query)

    def execute_query_with_cte(self, sql_query, prefix='T', return_df=False):
        if self.cte_count >0:
            traces_cte = [f"{k} as ({v})" for k,v in self.data['trace'].items()]
            execute_query = "WITH "+ ",".join(traces_cte)+ " "+ sql_query
        else:
            execute_query = sql_query
        self.generated_sql = execute_query
        res, description = self.execute_query( execute_query)
        new_cte_id = f"{prefix}_{self.prefix_tracker[prefix]}"
        columns = [i[0] for i in description]
        if return_df:
            return pd.DataFrame(res, columns =columns)
        self.data['trace'][new_cte_id]= sql_query
        self.cte[new_cte_id]= pd.DataFrame(res, columns =columns)
        self.prefix_tracker[prefix] += 1
        return new_cte_id
        
    def get_db(self):
        return get_db_schema(self.conn)


    def _get_schema(self):
        schema_list = list(read_sql_query("SELECT  sql FROM sqlite_master WHERE type='table';", self.conn)['sql'])
        return "## Schema\n\n"+"\n\n".join(schema_list)