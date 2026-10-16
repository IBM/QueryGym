from query_gym.dataset.unite_dataset import UniteDataset
from query_gym.envs import SQLiteQueryGym as QueryGym
import unittest
from dotenv import find_dotenv, load_dotenv
import os 

load_dotenv(find_dotenv())

def dataset_load(dataset):
    wiki_env = QueryGym(dataset=dataset)
    obs = wiki_env.reset()
    assert "Error" not in obs

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

class UniteTest(unittest.TestCase):
    
    def test_dataset_load(self):


        for i, j in datasets.items():
            print(f"Testing {i}")
            for mode in j:
                dataset = UniteDataset(
                    dataset_path=f"{UNITE_DATASET_PATH}/{i}",
                    mode=mode,
                )
                print(f"{i}: {mode}: {len(dataset)}")
                dataset_load(dataset)

    def test_example(self):
        dataset = UniteDataset(
                    dataset_path=f"{UNITE_DATASET_PATH}/cosql",
                    mode="dev",
                )
        env = QueryGym(dataset=dataset)
        obs = env.reset(1)
        from agents.gym_agent import GymnasiumAgent
        from langchain_openai import ChatOpenAI
        import yaml 
        import langchain 
        langchain.debug = True
        prompt_def  = yaml.safe_load(open("./query_gym/configs/action_definition.yaml"))
        
        available_actions = f"There are {len(env.ACTION_TYPES)} types of actions in this scenario. The action name, description, usage, and example is given below.\n" + "\n".join([f"{i}. {a}: {prompt_def['actions'][a.name]}" for i , a in enumerate(env.ACTION_TYPES)])
        
        llm = ChatOpenAI(
            model=os.environ.get("MODEL_ID"),
            temperature=0,
            max_retries=3,
            api_key='/',
            base_url=os.environ.get("LITELLM_BASE_URL"),
            # default_headers={}, 
        )
        agent = GymnasiumAgent(model=llm, instruction=prompt_def['system_instruction'].replace("{{get_actions}}", available_actions), max_msg_history=4)
        agent.reset()
        agent.observe(obs)
        from langchain_core.messages import AIMessage
        agent.message_history.append(AIMessage(content='Action: ["get_tables"]'))
        o, r,d,t,i = env.step(["get_tables"])
        agent.observe(o)
        action_str = agent.act()
        

if __name__ == '__main__':
    unittest.main()