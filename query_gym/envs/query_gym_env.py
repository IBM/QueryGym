import os
import random
import re
import traceback
from collections import defaultdict
from enum import StrEnum
from typing import Iterable, TypeVar

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from sqlgym.datasets import DbDataset

from ..utils.metrics import compare_result_dfs

T = TypeVar('T', int, str, float)
ParameterizedAction = Iterable[T]
 
ACTIONS = [ 'EXECUTE_SQL','GET_OVERVIEW', 'GET_QUERY', 'GET_SCHEMA', 'GET_TABLES', 'GET_COLUMNS','GET_ACTIONS', 'GET_OPERATIONS',  'PERFORM_PROJECTION', 'PERFORM_FILTER', 'PERFORM_JOIN', 'PERFORM_ORDER_BY', 'PERFORM_LIMIT', "PERFORM_AGGREGATE", "PERFORM_UNION", "PERFORM_INTERSECT",
          "PREVIEW_TABLE","GET_UNIQUE_VALUES","GET_COLUMN_TYPES", "GET_COLUMN_STATS","GET_SAMPLE_VALUES"]

TOOLS = ['COLUMN_ALIAS']
 

    
OPERATION_MAP = { k:k.lower() for k in ACTIONS if 'PERFORM' in k}

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

import re
from collections import defaultdict

from gymnasium.spaces import Text

class QueryGym(gym.Env):
    def __init__(self, dataset: DbDataset, action_definition=None, allow_execute_sql=False, insert_schemacontent_linking=False, GOAL_REWARD = 100, max_steps=200, STEP_REWARD=20, logger=None, **kwargs):
        self.idx = None
        self.dataset = dataset
        self.connection_kwargs = kwargs
        self.conn = None
        self.observation = "Call reset() to initialize the environment."
        if action_definition is None:
            with open(os.path.join(os.path.dirname(__location__), 'configs/action_definition.yaml'), "r") as f:
                action_definition = yaml.safe_load(f)['actions']
            # print(action_definition)
        self.ad = action_definition
        self.data = {}
        self.allow_execute_sql = allow_execute_sql
        self.prefix_tracker = defaultdict(int)
        self.logger = logger
        self.GOAL_REWARD = GOAL_REWARD
        if allow_execute_sql:
            self.ACTION_TYPES = StrEnum('ACTION_TYPES', {k: k.lower() for k in ACTIONS})
        else:
            self.ACTION_TYPES = StrEnum('ACTION_TYPES', {k: k.lower() for k in ACTIONS[1:]})
        self.STEP_REWARD = STEP_REWARD
        self.max_steps = max_steps
        self.observation_space = Text(max_length=10^5)
        
    def get_db_connection(self, path=None, **kwargs):
        raise NotImplementedError
    
    def get_db(self, conn=None, **kwargs):
        raise NotImplementedError
    
    def close_db_connection(self, conn=None, **kwargs):
        raise NotImplementedError
    
    def get_tables(self, **kwargs):
        raise NotImplementedError

    def get_gt(self):
        raise NotImplementedError
            
    def execute_query(self, query):
         raise NotImplementedError

    def execute_query_with_cte(self, sql_query, prefix='T', return_df=False):
        raise NotImplementedError
    
    def process_aliases(self, tables):
        parsed_tables = []
        for table in tables.split(","):
            # Look for " AS " (case-insensitive)
            m = re.split(r'\s+as\s+', table, flags=re.IGNORECASE)
            if len(m) == 2:
                table_name, alias = m[0].strip(), m[1].strip()
            else:
                table_name = table.strip()
                alias = table_name
            parsed_tables.append((table_name, alias))

        # If project_columns is not provided, build projection columns for each table.
        projection_list = []
        for table_name, alias in parsed_tables:
            # Create alias qualified column projections.
            projection_list.extend([f"{alias}.{col} as {alias}_{col}" for col in self.cte[table_name].columns])
        project_columns = ", ".join(projection_list)
        return project_columns

    def reset(  
        self,
        idx=None,
        seed=None,
        options=None,
    ) -> str:
        if idx == None:
            random.seed(seed)
            idx = random.randrange(len(self.dataset))
        super().reset(seed=seed, options=options)
        self.idx = idx
        self.dataset_item = self.dataset[self.idx]
        self.data = self.dataset._data[self.idx]
        self.data['path'] =  self.dataset_item.path
        self.data['idx'] = idx
        self.data['steps'] = 0
        self.data['GT'] = self.dataset_item.gt
        self.data['operations'] = 0
        self.data['operation_fail'] = 0
        self.data['trace'] = {}
        self.conn = self.get_db_connection(**self.connection_kwargs)
        if hasattr(self.dataset, "tables"):
            self.db = self.dataset.tables[self.data['db_id']]
        else:
            self.db = self.get_db()
            self.db['db_id'] = self.data['db_id']
        self.generated_sql = ""
        self.prefix_tracker = defaultdict(int)
        self.cte =  self.get_tables()
        self.gt = self.get_gt()
        return self._get_observation()


    def _get_observation(self):
        obs = f"The database `{self.data['db_id']}` contains following tables: "+ ",".join(self.db['table_names_original'])+". \n" 
        for tbl in self.db['table_names_original']:
            obs += f"Table `{tbl}` contains following columns: `"+ ",".join(list(self.cte[tbl].columns)) +"`. \n" 
        obs += "User Query: "+ self.data['question']
        return obs
        

    def step(self, action: ParameterizedAction ):
        self.data['steps'] += 1
        action_type, action_params = action[0].lower(), action[1:]
        reward, done, truncated, info = 0, False, False, self.data.copy()
        if self.max_steps <= self.data['steps']:
            truncated = True
        info['generated_sql'] = ""
        info['operation_success'] = None
        info.update({"match": 0, "non_empty_match":0, "subset_match": 0})
        if "perform" in action_type:
            self.data['operations'] += 1
            operation, operation_params = action_type, action[1:]
            operation_success = False 
            table_id_or_msg = "No Such Operation Found"
            if operation == self.ACTION_TYPES.PERFORM_PROJECTION:
                table_id_or_msg, operation_success = self._perform_projection(*operation_params)
            elif operation == self.ACTION_TYPES.PERFORM_FILTER:
                table_id_or_msg, operation_success = self._perform_filter(*operation_params)
            elif operation == self.ACTION_TYPES.PERFORM_ORDER_BY:
                table_id_or_msg, operation_success = self._perform_orderby(*operation_params)
            elif operation == self.ACTION_TYPES.PERFORM_LIMIT:
                table_id_or_msg, operation_success = self._perform_limit(*operation_params)
            elif  operation == self.ACTION_TYPES.PERFORM_JOIN:
                table_id_or_msg, operation_success = self._perform_join(*operation_params)
            elif  operation == self.ACTION_TYPES.PERFORM_AGGREGATE:
                table_id_or_msg, operation_success = self._perform_aggregate(*operation_params)
            elif  operation == self.ACTION_TYPES.PERFORM_UNION:
                    table_id_or_msg, operation_success = self._perform_union(*operation_params)
            elif  operation == self.ACTION_TYPES.PERFORM_INTERSECT:
                    table_id_or_msg, operation_success = self._perform_intersect(*operation_params)
            info['generated_sql'] = self.generated_sql
            info['operation_success'] = operation_success
            if not operation_success:
                self.data['operation_fail'] += 1
                # self.logger.log(table_id_or_msg)
                return table_id_or_msg, reward, done, truncated, info
            reward, done, reward_info = self._compute_reward_and_terminated(table_id_or_msg)
            info.update(reward_info)
            return self._summarize_table(table_id_or_msg), reward, done, truncated, info 
        elif action_type == self.ACTION_TYPES.GET_SCHEMA:
            return self._get_schema(),  reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_OVERVIEW:
            return self._get_observation(),  reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_QUERY:
            return self.data['question'],  reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_TABLES:
            return self._get_tables(),  reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_COLUMNS:
            return self._get_columns(action_params[0]),  reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_OPERATIONS:
            return self._get_operations(), reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_ACTIONS:
            return self.get_actions(), reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.PREVIEW_TABLE:
            return self._preview_table(action_params[0]), reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_UNIQUE_VALUES:
            return self._get_unique_values(*action_params), reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_SAMPLE_VALUES:
            return self._get_sample_values(*action_params), reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_COLUMN_TYPES:
            return self._get_columns_type(*action_params),  reward, done, truncated, info
        elif action_type == self.ACTION_TYPES.GET_COLUMN_STATS:
            return self._get_column_stats(*action_params), reward, done, truncated, info
        elif "execute_sql"==action_type and self.allow_execute_sql:
            try: 
                table_id_or_msg = self.execute_query_with_cte(action_params[0])
                info['generated_sql'] = self.generated_sql
                # self.logger.log("generated_sql",  self.generated_sql)
            except Exception as e:
                # logging the error message
                table_id_or_msg =  f"Execute SQL failed.\n\n\n# Exception\n\n```\n{traceback.format_exc()}\n```\n"
                info['generated_sql'] = self.generated_sql
                # self.logger.log("generated_sql",  self.generated_sql)
                return table_id_or_msg, reward, done, truncated, info
            reward, done, reward_info = self._compute_reward_and_terminated(table_id_or_msg)
            info.update(reward_info)
            return self._summarize_table(table_id_or_msg), reward, done, truncated, info
        else:
            return "Unrecognized action. "+self.get_actions(), reward, done, truncated, info

    @property
    def cte_count(self):
        if len(self.prefix_tracker) == 0:
            return 0
        return sum(self.prefix_tracker.values())

    def _compute_reward_and_terminated(self, table_id):
        # TODO: Allow different Rewards
        # if self.gt.values.shape != self.cte[table_id].values.shape:
        #     return 0, False
        # if (self.gt.values == self.cte[table_id].values).all():
        #     return self.GOAL_REWARD, True
        # if self.compare_dfs_ignore_colnames_unordered_rows(self.gt, self.cte[table_id]):
        #     return self.GOAL_REWARD, True
        # if self.compare_dfs_ignore_colnames_subset(self.gt, self.cte[table_id]):
        #     return self.STEP_REWARD, True
        # if self.is_subset_ignore_colnames(self.gt, self.cte[table_id]):
        #     return self.STEP_REWARD, True
        # # Following code compares column names as well.
        # return self.cte[table_id].equals(self.gt)
        match, non_empty_match, subset_match,row_correctness = compare_result_dfs(self.gt, self.cte[table_id], self.dataset_item.gt)
        reward = 0
        done = False
        if match:
            reward += self.GOAL_REWARD
            done = True
            subset_match = 1
        elif non_empty_match:
            reward += self.GOAL_REWARD
            done = True
            subset_match = 1
        elif subset_match:
            reward += self.STEP_REWARD
        return reward, done, {"match": match, "non_empty_match":non_empty_match, "subset_match": subset_match, "row_length_correctness": row_correctness}

    def get_column_id(self, table_id, column_id ):
        column_id = column_id.replace('`',"")
        if f"{table_id}." in column_id:
            return column_id.replace(f"{table_id}.","")
        return column_id
      
    def _get_sample_values(self, table_id, column_id):
        table_id = table_id.replace('`',"")
        column_id = self.get_column_id(table_id, column_id)
        if table_id not in self.cte:
            if table_id not in self.db['table_names']:
                return f"Error: Table {table_id} does not exists.\n" + self._get_tables()
            else:
                table_id = self.db['table_names_original'][self.db['table_names'].index(table_id)]
        if 10 > len(self.cte[table_id][column_id]):
            sample_values = self.cte[table_id][column_id]
            return f"The column {column_id} contains {len(sample_values)} rows shown below. \n{sample_values.to_markdown()}"
        else:
            sample_values = self.cte[table_id][column_id].sample(10)
            return f"The column `{column_id}` contains {len(self.cte[table_id][column_id])} rows. A sample of 10 values are shown below. \n{sample_values.to_markdown()}"

    def _get_unique_values(self, table_id, column_id):
        table_id = table_id.replace('`',"")
        column_id = self.get_column_id(table_id, column_id)
        if table_id not in self.cte:
            if table_id not in self.db['table_names']:
                return f"Error: Table {table_id} does not exists.\n" + self._get_tables()
            else:
                table_id = self.db['table_names_original'][self.db['table_names'].index(table_id)]
        
        unique_values = self.cte[table_id][column_id].unique()
        if 10 > len(unique_values):
            return f"The column `{column_id}` contains {len(unique_values)} unique values shown below. \n{str(unique_values)}"
        else:
            sample_values = pd.Series(unique_values).sample(10)
            return f"The column `{column_id}` contains {len(unique_values)}  unique values. A sample of 10 values are shown below. \n{sample_values.to_markdown()}"
        
    def _preview_table(self, table_id):
        table_id = table_id.replace('`',"")
        if table_id not in self.cte:
            if table_id not in self.db['table_names']:
                return f"Error: Table {table_id} does not exists.\n" + self._get_tables()
            else:
                table_id = self.db['table_names_original'][self.db['table_names'].index(table_id)]
        table_md = self.cte[table_id].head().to_markdown()
        nrows, ncolumns = self.cte[table_id].shape
        return f"Table `{table_id}` has {nrows} rows and {ncolumns} columns. The first 5 rows are shown below.\n{table_md}"
    
    def _get_tables(self):
        tables = " \n".join([ f"{n+1}. `{i}` " for n, i in enumerate(self.db['table_names_original']) ])
        return_text= f"This database has {len(self.db['table_names_original'])} tables. The table IDs used to reference them are listed below.\n{tables}"
        if self.cte_count > 0:
            return_text += f"\nThere are {self.cte_count} temporary result sets, also known as CTEs. Each CTE may builds upon the result of the previous CTE or the tables in the database. Below are the CTE names along with their corresponding definitions:\n"+ "\n".join([f"{k}: {v}" for k, v in self.data['trace'].items()])
        return return_text
    
    def _get_schema(self):
        raise NotImplementedError

    def _get_columns_type(self, table_id):
        # TODO : Revisit for different datasets; current implementation is BIRDSQL specific
        if table_id in self.db['table_names_original'] or table_id in self.db['table_names']:
            table_idx = None
            if table_id in self.db['table_names_original']:
                table_idx = self.db['table_names_original'].index(table_id)
            else:
                table_idx = self.db['table_names'].index(table_id)
            columns =[ f"{n}. {i[1]}: {k}" for n, (i,  k) in enumerate(zip(self.db['column_names_original'],  self.db['column_types'])) if i[0] == table_idx ]
            return f"Table {self.db['table_names_original'][table_idx]} has {len(columns)} columns. The column indexes, names, and their data types are listed below.\n" + " \n".join(columns)
        elif table_id in self.cte.keys():
            column_type = self.cte[table_id].dtypes.to_markdown()
            return f"Table {table_id} is a temporary result set, commonly known as Common Table Expression (CTE). The exact SQL defintion of {table_id} is:\n```\n{self.data['trace'][table_id]}\n```\n It has {len(self.cte[table_id].columns)} columns. The column names and their types are listed below.\n" + column_type
        else:
            return f"Table {table_id} was not found." + self._get_tables()
    
    def _get_column_stats(self, table_id, column_id):
        table_id = table_id.replace('`',"")
        column_id = self.get_column_id(table_id, column_id)
        if table_id not in self.cte:
            if table_id not in self.db['table_names']:
                return f"Error: Table {table_id} does not exists.\n" + self._get_tables()
            else:
                table_id = self.db['table_names_original'][self.db['table_names'].index(table_id)]
        md_column = self.cte[table_id][column_id].describe(include="all").to_markdown()       
        return f"The summary statistics for {column_id} in table {table_id} is shown below.\n{md_column}."
    
    
    def _get_columns(self, table_id):
        table_id = table_id.replace('`',"")
        table_idx = None
        if table_id in self.db['table_names_original'] or table_id in self.db['table_names']:
            if table_id in self.db['table_names_original']:
                table_idx = self.db['table_names_original'].index(table_id)
            else:
                table_idx = self.db['table_names'].index(table_id)
            columns =[ f"{n}. {i[1]} --- {j[1]}" for n, (i, j) in enumerate(zip(self.db['column_names_original'], self.db['column_names'])) if i[0] == table_idx ]
            return f"Table {self.db['table_names_original'][table_idx]} has {len(columns)} columns. The column indexes, names, and descriptions are listed below.\n" + " \n".join(columns)
        elif table_id in self.cte:
            columns = self.cte[table_id].columns
            return f"Table {table_id} is a temporary result set, commonly known as Common Table Expression (CTE). The exact SQL defintion of {table_id} is:\n```\n{self.data['trace'][table_id]}\n```\n It has {len(columns)} columns. The column names are listed below.\n" + " \n".join(columns)
        else:
            return f"Table {table_id} was not found." + self._get_tables()
    
    def get_actions(self):
        return f"There are {len(self.ACTION_TYPES)} types of actions in this scenario. The action name, description, usage, and example is given below.\n" + "\n".join([f"{i+1}. {a}: {self.ad[a.name]}" for i , a in enumerate(self.ACTION_TYPES)])
        
    def _get_operations(self):
        return f"{len(OPERATION_MAP)} operations are allowed on this database. The operation index, name, description, and example is given below.\n" +  "\n".join([f"{i}. {a}: {self.ad[a]}" for i , a in enumerate(OPERATION_MAP)])
    
    def _perform_projection(self, table, columns, prefix='T'):
        try:
            sql_query = f"SELECT {columns} from {table}"
            return self.execute_query_with_cte(sql_query, prefix), True
        except Exception as e:
            return f"Projection operation failed. Table was {table}, and columns were {columns}. \n\n# Exception\n\n```\n{traceback.format_exception_only(e)[0]}\n```\n", False

    def _perform_aggregate(self, table, group_column, select_columns, having_condition=None, prefix='T'):
        try:
            if having_condition is None or len(having_condition) == 0:
                sql_query = f"SELECT {select_columns} from {table} GROUP BY {group_column}"
            else:
                sql_query = f"SELECT {select_columns} from {table} GROUP BY {group_column} HAVING {having_condition}"

            return self.execute_query_with_cte(sql_query, prefix), True
        except Exception as e:
            return f"Aggregate operation failed. Table was {table}, group column was {group_column}, and select columns were {select_columns}. \n\n# Exception\n\n```\n{traceback.format_exception_only(e)[0]}\n```\n", False

    def _perform_union(self, all_or_distinct, table1, table2, table1_cols='*', table2_cols="*", prefix='T'):
        try:
            if all_or_distinct.lower() == "distinct":
                all_or_distinct=""
            sql_query = f"SELECT {table1_cols} from {table1} UNION {all_or_distinct} SELECT {table2_cols} from {table2}"
            return self.execute_query_with_cte(sql_query, prefix), True
        except Exception as e:
            return f"Union operation failed. SQL query was {sql_query}. \n\n# Exception\n\n```\n{traceback.format_exception_only(e)[0]}\n```\n", False

    def _perform_intersect(self, table1, table2, table1_cols='*', table2_cols="*", prefix='T'):
        try:
            sql_query = f"SELECT {table1_cols} from {table1} INTERSECT SELECT {table2_cols} from {table2}"
            return self.execute_query_with_cte(sql_query, prefix), True
        except Exception as e:
            return f"Intersect operation failed.\n The SQL query was as follows: \n ```\n{sql_query}\n```\n\n\n# Exception\n\n```\n{traceback.format_exception_only(e)[0]}\n```\n", False

    def _perform_filter(self, table, condition, project_columns="*", prefix='T'):
        try:
            sql_query = f"SELECT {project_columns} FROM {table} WHERE {condition}"
            return self.execute_query_with_cte(sql_query, prefix), True
        except Exception as e:
            return f"Filter operation failed. Table was {table} and condition was {condition}. The project_columns  was {project_columns}.\n \n\n# Exception\n\n```\n{traceback.format_exception_only(e)[0]}\n```\n", False
    
    def _perform_orderby(self, table, condition, project_columns="*", prefix='T'):
        try:
            sql_query = f"SELECT {project_columns} FROM {table} ORDER BY {condition}"
            return self.execute_query_with_cte(sql_query, prefix), True
        except Exception as e:
            return f"OrderBy operation failed. Table was {table} and condition was {condition}. \n\n# Exception\n\n``` \n{traceback.format_exception_only(e)[0]}\n```\n", False
   
    def _perform_limit(self, table, condition, project_columns="*", prefix='T'):
        try:
            sql_query = f"SELECT {project_columns} FROM {table} LIMIT {condition}"
            return self.execute_query_with_cte(sql_query, prefix), True
        except Exception as e:
            return f"Limit operation failed. Table was {table} and condition was {condition}. \n\n# Exception\n\n```\n{traceback.format_exception_only(e)[0]}\n```\n", False     
    
    def _perform_join(self, tables, join_conditions, join_types,  project_columns=None, prefix='T'):
        import re
        try:

            # Parse each table to extract the table name and alias.
            # If no alias is provided, use the table name as its alias.
            parsed_tables = []
            for table in tables:
                # Look for " AS " (case-insensitive)
                m = re.split(r'\s+as\s+', table, flags=re.IGNORECASE)
                if len(m) == 2:
                    table_name, alias = m[0].strip(), m[1].strip()
                else:
                    table_name = table.strip()
                    alias = table_name
                parsed_tables.append((table_name, alias))

            # If project_columns is not provided, build projection columns for each table.
            if project_columns is None:
                projection_list = []
                for table_name, alias in parsed_tables:
                    # Create alias qualified column projections.
                    projection_list.extend([f"{alias}.`{col}` as {alias}_{col}" for col in self.cte[table_name].columns])
                project_columns = ", ".join(projection_list)

            # Build the JOIN query.
            sql_query = f"SELECT {project_columns} FROM {tables[0]}"
            # print(sql_query)
            # For each join, chain the join type, next table, and join condition.
            for i in range(len(join_conditions)):
                sql_query += f" {join_types[i]} {tables[i+1]} ON {join_conditions[i]}"
            
            # logger.log(sql_query)

            return self.execute_query_with_cte(sql_query, prefix), True
        except Exception as e:
            return f"Join operation failed. Joining tables {tables} with conditions {join_conditions} and projected columns {project_columns}. \n\n# Exception\n\n```\n{traceback.format_exception_only(e)[0]}\n```\n", False

    def _describe_column(self, table_id, column_id):
        # TODO: Check if we want to add more details in the observation
        columns = self.cte[table_id][column_id].describe().to_markdown()
        
        if len(columns) == 1 and len(self.cte[table_id]) == 1:
            # If only one single cell, return the value
            desc = f"Resulting table, {table_id}, has single value of {str(self.cte[table_id].iloc[0,0])}"
        else:
            desc = f"Table {table_id} has {len(columns)} column(s) and {len(self.cte[table_id])} row(s). "  
            desc += "Column names are "+",".join(columns) +"."
        return desc  
        
    def _summarize_table(self, table_id):
        # TODO: Check if we want to add more details in the observation
        columns = self.cte[table_id].columns
        if len(columns) == 1 and len(self.cte[table_id]) == 1:
            # If only one single cell, return the value
            desc = f"Resulting table, {table_id}, has single value of {str(self.cte[table_id].iloc[0,0])}"
        else:
            desc = f"Table {table_id} has {len(columns)} column(s) and {len(self.cte[table_id])} row(s).\n"  
            desc += "Column names are "+",".join(columns) +"."
            # desc += self.cte[table_id].head(10).to_markdown()
            desc += "\n"
        return desc
        

    
    def close(self):
        self.close_db_connection()
        return super().close()