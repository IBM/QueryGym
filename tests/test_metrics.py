from query_gym.utils.metrics import compare_result_dfs, compare_dfs_ignore_colnames_ordered_rows, compare_dfs_ignore_colnames_subset_slow, compare_dfs_ignore_colnames_subset3
import pandas as pd
from query_gym.envs import SQLiteQueryGym as QueryGym
import unittest
import numpy as np
from query_gym.dataset.unite_dataset import UniteDataset

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


UNITE_DATASET_PATH ="/data/datasets/unified/"

class MetricTest(unittest.TestCase):
    
    def test_subset_difforder(self):
        
        df1 = pd.DataFrame([3, 6,7, 2,1], columns=['a'])
        df2 = pd.DataFrame([[1, 1, 3],
                        [2, 1, 1],
                        [3, 8, 8],
                        [6, 0, 1],
                        [7, 1, 0]], columns=[ 'a','b','c'])
        # compare_dfs_ignore_colnames_subset_slow and
        # but compare_dfs_ignore_colnames_subset fails
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, True)
        assert subset_match
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, False)
        assert subset_match == False
        
                
        df2 = pd.DataFrame([[ 1, 3, 1],
                        [ 1, 1, 2],
                        [8, 8, 3],
                        [0, 1, 6],
                        [1, 0, 7]], columns=[ 'a','b','c'])
        # compare_dfs_ignore_colnames_subset_slow and
        # but compare_dfs_ignore_colnames_subset fails
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, True)
        assert subset_match
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, False)
        assert subset_match == False

    def test_subset_multirow(self):
        
        df1 = pd.DataFrame([[1, 4],
                            [3, 6],
                            [2, 5] ], columns=['a','b'])
        df2 = pd.DataFrame([ [2, 4],
                            [1, 5],
                            [3, 6] ], columns=[ 'd','c'])
        # compare_dfs_ignore_colnames_subset_slow is working 
        # but compare_dfs_ignore_colnames_subset is not working for this case
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, False)
        assert subset_match   == False
        
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, True)
        # assert subset_match == False
        # gives False positive 

    def test_subset(self):
        
        df1 = pd.DataFrame([1,2,3,6,7], columns=['a'])
        df2 = pd.DataFrame([[1, 1, 3],
                        [2, 1, 1],
                        [3, 8, 8],
                        [6, 0, 1],
                        [7, 1, 0]], columns=[ 'a','b','c'])
        sql = "select a from table"
        # compare_dfs_ignore_colnames_subset_slow is working 
        # but compare_dfs_ignore_colnames_subset is not working for this case
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, True)
        assert subset_match
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, False)
        assert subset_match  
        
        df1 = pd.DataFrame([1,2,3,6,7], columns=['a'])                 
        df2 = pd.DataFrame([[ 1, 3, 1],
                        [ 1, 1, 2],
                        [8, 8, 3],
                        [0, 1, 6],
                        [1, 0, 7]], columns=[ 'a','b','c'])
        
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, True)
        assert subset_match
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, False)
        assert subset_match  


        
    def test_mix_type(self):
        df1 = pd.DataFrame([[45.0],
                        ['east'],
                        ['east'],
                        ['east']], columns=['a'])    
        df2 = pd.DataFrame([[45.0],
                        ['east'],
                        ['east'],
                        ['east']], columns=['a'])    
        subset_match = compare_dfs_ignore_colnames_subset3(df1, df2, True)
        assert subset_match 
        
        
    def test_mix_type_bird(self):
        train_dataset =  UniteDataset(
            dataset_path=f'{UNITE_DATASET_PATH}/bird_train/',
            mode="train",
        )
        train_env = QueryGym(dataset=train_dataset)

        train_env.reset(728)
        o, r, d, t, i = train_env.step(['perform_aggregate','cars', 'train_id', 'train_id, MAX(position) AS trailPosi'])
        o, r, d, t, i = train_env.step(['perform_join',['T_0 as temp_alias', 'trains'], ["trains.id = temp_alias.train_id and temp_alias.trailPosi >= 4"], ['INNER JOIN'], "trains.direction" ])
        o, r, d, t, i = train_env.step(['perform_join',['T_0 as T2', 'trains as T1'], ["T1.id = T2.train_id"], ['INNER JOIN'], " CAST(COUNT(CASE WHEN T2.trailPosi >= 4 THEN T1.id ELSE NULL END) AS REAL) * 100 / COUNT(T1.id)" ])
        o, r, d, t, i = train_env.step(['perform_union','ALL','T_2', 'T_1'])
        assert r == 100
        assert d == True
        o, r, d, t, i = train_env.step(['perform_intersect','T_2', 'T_1'])
        assert "has 1 column(s) and 0 row(s)" in o

    def test_bird_example3(self):

        dataset = UniteDataset(
            dataset_path=f"{UNITE_DATASET_PATH}/bird_train",
            mode="train",
        )
        env = QueryGym(dataset=dataset)
        obs = env.reset(3)
        o, r, d, t, i = env.step(["perform_group_by", "rating", "", "MAX(SUM(rating_score))"])
        o, r, d, t, i = env.step(["perform_order_by", "T_0", "movie_popularity DESC", "movie_title, movie_popularity"])
        # Subset Match is currently for Column Subset, not for rows?
        assert r == 0
        
        
    def test_bird_example0(self):

        dataset = UniteDataset(
            dataset_path=f"{UNITE_DATASET_PATH}/bird_train",
            mode="train",
        )
        env = QueryGym(dataset=dataset)
        obs = env.reset(0)
        o, r, d, t, i = env.step(["perform_filter", "movies", "movies.movie_release_year = 1945"])
        o, r, d, t, i = env.step(["perform_order_by", "T_0", "movie_popularity DESC", "movie_title, movie_popularity"])
        # Subset Match is currently for Column Subset, not for rows?
        assert r == 0
        
    def test_criteria_example1(self):

        dataset = UniteDataset(
            dataset_path=f"{UNITE_DATASET_PATH}/Criteria2SQL",
            mode="train",
        )
        env = QueryGym(dataset=dataset)
        obs = env.reset(0)
        o, r, d, t, i = env.step(["perform_filter", "records", "age > 18 AND (severe_sepsis = 1 OR septic_shock = 1)"])
        assert r == env.STEP_REWARD

    def test_cosql_example1(self):

        dataset = UniteDataset(
            dataset_path=f"{UNITE_DATASET_PATH}/cosql",
            mode="train",
        )
        env = QueryGym(dataset=dataset)
        obs = env.reset(1)
        o, r, d, t, i = env.step(['perform_aggregate', 'instructor', 'dept_name', 'dept_name, AVG(salary) as average_salary'])
        assert r == env.GOAL_REWARD
        obs = env.reset(1)
        o, r, d, t, i = env.step(["perform_aggregate", "instructor", "dept_name", "dept_name, AVG(`salary`)", ""])
        assert r == env.GOAL_REWARD

if __name__ == '__main__':
    unittest.main()