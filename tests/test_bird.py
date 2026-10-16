from query_gym.envs import SQLiteQueryGym as QueryGym
from query_gym.dataset.unite_dataset import UniteDataset
import unittest
import tracemalloc

try:
    import psutil
except ImportError as e:
    print("psutil package not installed")

dataset_path = "/data/datasets/unified"

class BirdTest(unittest.TestCase):
    
    def test_issue_28(self):       
        dataset = UniteDataset(
            dataset_path=f'{dataset_path}/bird_dev',
            mode="dev",
        )
        env = QueryGym(dataset=dataset)
        obs = env.reset(1533)
        self.assertIsNotNone(obs, f"Example  failed")
        o, r, d, t, i = env.step(['GET_SCHEMA'])
        print(o)
        
    
    def test_dev_load(self):
        dataset = UniteDataset(
            dataset_path=f'{dataset_path}/bird_dev/',
            mode="dev",
        )
        # for i in range(len(dataset)):
        for i in range(10, 20):
            env = QueryGym(dataset=dataset)
            obs = env.reset(i)
            process = psutil.Process()
            print(f"Epsiode {i}: {process.memory_info().rss / 1024 ** 2}")
            self.assertIsNotNone(obs, f"Example {i} failed")
            env.close()
    
    def test_get_operations(self):
        
        dataset = UniteDataset(
            dataset_path=f'{dataset_path}/bird_dev',
            mode="dev",
        )
        env = QueryGym(dataset=dataset)
        obs= env.reset(0)
        o, _ , _ , _ , _ = env.step(["get_overview"])
        assert "Error" not in o
        o, _ , _ , _ , _ = env.step(["get_actions"])
        assert "Error" not in o
        o, _ , _ , _ , _ = env.step(["get_operations"])
        assert "Error" not in o
        o, _ , _ , _ , _ = env.step(["get_query"])
        assert dataset._data[0]['question'] in o
        o, _ , _ , _ , _ = env.step(["get_tables"])
        assert "3" in o
        o, _ , _ , _ , _ = env.step(["perform_filter", "frpm", "`County Name` == 'Alameda'"])
        assert "T_0" in o
        o, _ , _ , _ , _ = env.step(["get_columns", "frpm"])
        assert "29 columns" in o
        o, _ , _ , _ , _ = env.step(["get_columns", "T_0"])
        assert "29 columns" in o
        o, _ , _ , _ , _ = env.step(["preview_table", "T_0"])
        assert "first 5 rows" in o
        assert "Error" not in o
        o, _ , _ , _ , _ = env.step(["preview_table", "frpm"])
        assert "first 5 rows" in o
        assert "Error" not in o
        o, _ , _ , _ , _ = env.step(["get_column_stats", "T_0", "County Name"])
        assert "count  | 379" in o
        o, _ , _ , _ , _ = env.step(["get_column_stats", "frpm", "County Name"])
        assert "count  | 9986" in o
        o, _ , _ , _ , _ = env.step(["get_unique_values", "T_0", "County Name"])
        assert "1 unique values" in o
        o, _ , _ , _ , _ = env.step(["get_unique_values", "frpm", "County Name"])
        assert "58" in o and "sample" in o
        o, _ , _ , _ , _ = env.step(['perform_projection', "T_0", "`Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`"])
        o, _ , _ , _ , _ = env.step(["get_column_types", "frpm"])
        assert "29 columns" in o
        o, _ , _ , _ , _ = env.step(["get_column_types", "T_1"])
        assert "Error" not in o
        o, _ , _ , _ , _ = env.step(["get_sample_values", "frpm","County Code"])
        assert "Error" not in o
        env.step(['perform_projection', "T_0", "`Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` as rate"])
        o, _ , _ , _ , _ = env.step(["get_sample_values", "T_2","rate"])
        assert "Error" not in o
        
    def test_dev_samples(self):
        dataset = UniteDataset(
            dataset_path=f'{dataset_path}/bird_dev',
            mode="dev",
        )
        env = QueryGym(dataset=dataset)

        # # "question_id": 0
        # """
        # {
        #         "question_id": 0,
        #         "db_id": "california_schools",
        #         "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        #         "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
        #         "SQL": "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
        #         "difficulty": "simple"
        #     },
        # ```
        # WITH 
        # T0 as (Select * from frpm WHERE `County Name` == 'Alameda'),     % -------- Filter
        # T1 as (SELECT * FROM T0 ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC),        % ---- orderby
        # T2 as (SELECT * from T1 LIMIT 1)    % -------- limit
        # SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` from T2 % -------- Project 
        # ```
        # """

        obs= env.reset(0)
        o, r, d, t, i = env.step(["perform_filter", "frpm", "`County Name` == 'Alameda'"])
        o, r, d, t, i = env.step(['perform_order_by', "T_0", "(CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC"])
        o, r, d, t, i = env.step(['perform_limit', "T_1", "1"])
        o, r, d, t, i = env.step(['perform_projection', "T_2", "`Free Meal Count (K-12)` / `Enrollment (K-12)`"])
        assert r == 100
        assert d == True


        # # "question_id": 1

        # """
        #     {
        #         "question_id": 1,
        #         "db_id": "california_schools",
        #         "question": "Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.",
        #         "evidence": "Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`",
        #         "SQL": "SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3",
        #         "difficulty": "moderate"
        #     }


        # ```
        # WITH
        # T0 as (SELECT * FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL),
        # T1 as (SELECT * FROM T0    ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC),
        # T2 as (SELECT & FROM T1 LIMIT 3)
        # SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM T3
        # ```
        # """

        obs= env.reset(1)
        o, r, d, t, i = env.step(["perform_filter", "frpm", "`Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL"])
        o, r, d, t, i = env.step(['perform_order_by', "T_0", "`Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC"])
        o, r, d, t, i = env.step(['perform_limit', "T_1", "3"])
        o, r, d, t, i = env.step(['perform_projection', "T_2", "`Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`"])
        assert r == 100
        assert d == True


        # # "question_id": 531

        # """
                            
        # {
        #         "question_id": 531,
        #         "db_id": "codebase_community",
        #         "question": "Which user has a higher reputation, Harlan or Jarrod Dixon?",
        #         "evidence": "\"Harlan\" and \"Jarrod Dixon\" are both DisplayName; highest reputation refers to Max(Reputation)",
        #         "SQL": "SELECT DisplayName FROM users WHERE DisplayName IN ('Harlan', 'Jarrod Dixon') AND Reputation = ( SELECT MAX(Reputation) FROM users WHERE DisplayName IN ('Harlan', 'Jarrod Dixon') )",
        #         "difficulty": "simple"
        #     }
            
        # ```

        # WITH
        # T0 as (SELECT * FROM users WHERE DisplayName IN ('Harlan', 'Jarrod Dixon')),
        # T1 as (SELECT MAX(Reputation) FROM T1),
        # T2 as (SELECT * FROM users WHERE DisplayName IN ('Harlan', 'Jarrod Dixon') AND Reputation = 457)
        # SELECT DisplayName FROM T2)
        # ```

        # """ 


        obs= env.reset(531)
        o, r, d, t, i = env.step(['perform_filter', "users", "DisplayName IN ('Harlan', 'Jarrod Dixon')"])
        o, r, d, t, i = env.step(['perform_projection', "T_0", "MAX(Reputation)"])
        o, r, d, t, i = env.step(['perform_filter', "users", "DisplayName IN ('Harlan', 'Jarrod Dixon') AND Reputation = 457"])
        o, r, d, t, i = env.step(['perform_projection', "T_2", "DisplayName"])
        assert r == 100
        assert d == True


        # """
        #     {
        #         "question_id": 537,
        #         "db_id": "codebase_community",
        #         "question": "How many posts does the user csgillespie own?",
        #         "evidence": "\"csgillespie\" is the DisplayName of user",
        #         "SQL": "SELECT COUNT(T1.id) FROM posts AS T1 INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id WHERE T2.DisplayName = 'csgillespie'",
        #         "difficulty": "simple"
        #     },
            
            
        #    SELECT count(posts.id) FROM posts INNER JOIN users ON posts.OwnerUserId = users.Id and users.DisplayName = 'csgillespie'
        # """

        obs= env.reset(537)
        o, r, d, t, i = env.step(['perform_join', ["posts", "users"], ["posts.OwnerUserId = users.Id and users.DisplayName = 'csgillespie'"],["INNER JOIN"], "count(posts.id)"])
        assert r == 100
        assert d == True



        """

            {
                "question_id": 1113,
                "db_id": "european_football_2",
                "question": "For the team \"Hannover 96\", what was its defence aggression class on 2015/9/10?",
                "evidence": "\"Hannover 96\" refers to team_long_name = 'Hannover 96'; on 2015/9/10 refers to date LIKE '2015-09-10%';",
                "SQL": "SELECT t2.chanceCreationShootingClass FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t1.team_long_name = 'Hannover 96' AND t2.`date` LIKE '2015-09-10%'",
                "difficulty": "moderate"
            }
            
            
        T0 as (SELECT ta.chanceCreationShootingClass FROM Team_Attributes as ta INNER JOIN Team as t ON t.team_api_id = ta.team_api_id and t.team_long_name = 'Hannover 96' AND ta.date LIKE '2015-09-10%')"],

        """


        obs= env.reset(1113)
        o, r, d, t, i = env.step(['perform_join', [ "Team_Attributes as ta", "Team as t"], ["t.team_api_id = ta.team_api_id and t.team_long_name = 'Hannover 96' AND ta.date LIKE '2015-09-10%'"], ["INNER JOIN"], "ta.chanceCreationShootingClass"])
        assert r == 100
        assert d == True


        """

        {
                "question_id": 1114,
                "db_id": "european_football_2",
                "question": "What was the average overall rating for Marko Arnautovic from 2007/2/22 to 2016/4/21?",
                "evidence": "average overall rating refers to avg(overall_rating); Marko Arnautovic refers to player_name = 'Marko Arnautovic'; from 2007/2/22 to 2016/4/21 refers to the first 10 characters of date BETWEEN '2007-02-22' and '2016-04-21'",
                "SQL": "SELECT CAST(SUM(t2.overall_rating) AS REAL) / COUNT(t2.id) FROM Player AS t1 INNER JOIN Player_Attributes AS t2 ON t1.player_fifa_api_id = t2.player_fifa_api_id WHERE t1.player_name = 'Marko Arnautovic' AND SUBSTR(t2.`date`, 1, 10) BETWEEN '2007-02-22' AND '2016-04-21'",
                "difficulty": "challenging"
            }
            
        SELECT CAST(SUM(pa.overall_rating) AS REAL) / COUNT(pa.id) FROM Player as p INNER JOIN Player_Attributes as pa ON p.player_fifa_api_id = pa.player_fifa_api_id AND p.player_name = 'Marko Arnautovic' AND SUBSTR(pa.date, 1, 10) BETWEEN '2007-02-22' AND '2016-04-21'
            
        """

        obs= env.reset(1114)
        o, r, d, t, i = env.step(['perform_join', ["Player as p", "Player_Attributes as pa"], ["p.player_fifa_api_id = pa.player_fifa_api_id AND p.player_name = 'Marko Arnautovic' AND SUBSTR(pa.date, 1, 10) BETWEEN '2007-02-22' AND '2016-04-21'"],["INNER JOIN"], "CAST(SUM(pa.overall_rating) AS REAL) / COUNT(pa.id)"])
        assert r == 100
        assert d == True



    #### Train Domain
    def test_train_load(self):
        train_dataset =  UniteDataset(
            dataset_path=f'{dataset_path}/bird_train',
            mode="train",
        )
        train_env = QueryGym(dataset=train_dataset)



        # for i in range(len(train_dataset)):
        for i in [1]:
            obs = train_env.reset(i)
            train_env.conn.close()
            print(f"episode {i}")
            assert obs is not None, f"Example {i} failed"
                
                
    def test_train_samples(self):
        train_dataset =  UniteDataset(
            dataset_path=f'{dataset_path}/bird_train',
            mode="train",
        )
        train_env = QueryGym(dataset=train_dataset)



        """
            {
                        "db_id": "book_publishing_company",
                        "question": "Which one is the cheapest business book?",
                        "evidence": "business books refers to type = 'business'; cheapest book refers to MIN(price)",
                        "SQL": "SELECT title FROM titles WHERE type = 'business' ORDER BY price LIMIT 1"
                },
            
            1 Read [titles], [], false
        1.1 Filter (type = business)
        1.1.1 Project [title]
        1.1.1.1 Sort [price ASC NULLS FIRST], true
        1.1.1.1.1 LocalLimit 1
        1.1.1.1.1.1 GlobalLimit 1

        """    

        obs= train_env.reset(213)
        o, r, d, t, i = train_env.step(['perform_filter', "titles", "type = 'business'"])
        o, r, d, t, i = train_env.step(['perform_order_by', "T_0", "price ASC NULLS FIRST"])
        o, r, d, t, i = train_env.step(['perform_limit', "T_1", "1"])
        o, r, d, t, i = train_env.step(['perform_projection', "T_2", "title"])
        assert r == 100
        assert d == True
            
            
        """

        {
                "question_id": 728,
                'db_id': 'trains',
        'question': 'What is the percentage of all the trains with at least 4 cars? List the directions of the said trains.',
        'evidence': 'at least 4 cars refers to trailPosi > = 4; calculation = MULTIPLY(DIVIDE(count(trailPosi > = 4 then id), count(id)), 100)',
        'SQL': 'SELECT CAST(COUNT(CASE WHEN T2.trailPosi >= 4 THEN T1.id ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM trains AS T1 INNER JOIN ( SELECT train_id, MAX(position) AS trailPosi FROM cars GROUP BY train_id ) AS T2 ON T1.id = T2.train_id 
        UNION ALL 
        SELECT T1.direction FROM trains AS T1 INNER JOIN ( SELECT train_id, MAX(position) AS trailPosi FROM cars t GROUP BY train_id ) AS T2 ON T1.id = T2.train_id AND T2.trailPosi >= 4',
        'steps': 0,
        'trace': [],
        'join_counter': 0
            }
            
            
            The corresponding logical plan is: 
        ------------
        1 Read [cars], [], false
        1.1 SubqueryAlias t
        1.1.1 Aggregate [train_id], [train_id, MAX(position) AS trailPosi]
        1.1.1.1 SubqueryAlias T2
        1.1.2 Read [trains], [], false
        1.1.2.1 SubqueryAlias T1
        1.1.2.1.1 Join Inner, ((T1.id = T2.train_id) AND (T2.trailPosi >= 4))
        1.1.2.1.1.1 Project [T1.direction]
        1.1.3 Aggregate [train_id], [train_id, MAX(position) AS trailPosi]
        1.1.3.1 SubqueryAlias T2
        1.1.3.1.1 Join Inner, (T1.id = T2.train_id)
        1.1.3.1.1.1 Project [Alias(((cast(COUNT(CASE WHEN T2.trailPosi >= 4 THEN T1.id ELSE NULL END) AS REAL) * 100 / COUNT(T1.id)), None)]
        1.1.3.1.1.1.1 Union false, false
        ------------


        """

        train_env.reset(728)
        o, r, d, t, i = train_env.step(['perform_aggregate','cars', 'train_id', 'train_id, MAX(position) AS trailPosi'])
        o, r, d, t, i = train_env.step(['perform_join',['T_0 as temp_alias', 'trains'], ["trains.id = temp_alias.train_id and temp_alias.trailPosi >= 4"], ['INNER JOIN'], "trains.direction" ])
        o, r, d, t, i = train_env.step(['perform_join',['T_0 as T2', 'trains as T1'], ["T1.id = T2.train_id"], ['INNER JOIN'], " CAST(COUNT(CASE WHEN T2.trailPosi >= 4 THEN T1.id ELSE NULL END) AS REAL) * 100 / COUNT(T1.id)" ])
        o, r, d, t, i = train_env.step(['perform_union','ALL','T_2', 'T_1'])
        assert r == 100
        assert d == True
        o, r, d, t, i = train_env.step(['perform_intersect','T_2', 'T_1'])
        assert "has 1 column(s) and 0 row(s)" in o

        
        """
                {'db_id': 'disney',
        'question': 'Name actors who voiced more than five Disney characters.',
        'evidence': 'Actors who voiced refer to voice-actor;',
        'SQL': "SELECT 'voice-actor' FROM `voice-actors` GROUP BY 'voice-actor' HAVING COUNT(movie) > 5",
        'steps': 0,
        'trace': [],
        'join_counter': 0}
        """
        train_env.reset(4733)
        o, r, d, t, i = train_env.step(['perform_aggregate', '`voice-actors`', '`voice-actor` HAVING Count(movie) > 5', '`voice-actor`'])
        assert r == 100
        assert d == True
            
            

if __name__ == '__main__':
    tracemalloc.start()
    unittest.main()