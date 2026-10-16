import json
from pathlib import Path
from typing import Dict, List
from sqlgym.datasets.utils import DbDataset, DbDatasetItem, SqlGymEnvModeEnum


class BirdCriticDatasetItem(DbDatasetItem):
    def __init__(self, gt, query, preprocess_steps, postprocess_steps, info,path):
        super().__init__(gt=gt, query=query, info=info,path=path)  # if these are needed by the parent
        self.gt = gt
        self.query = query
        self.preprocess_steps = preprocess_steps or []
        self.postprocess_steps = postprocess_steps or []
        self.info = info

class BirdCriticDataset(DbDataset):
    def __init__(self, dataset_path, mode='issue_with_gt'):
        self.path = Path(dataset_path)
        self.sql_gym_env_mode = SqlGymEnvModeEnum.SINGLE
        self.mode = mode
        with open(
            self.path.joinpath( f"{self.mode}.jsonl"),
            "r",
            encoding="utf8",
        ) as f:
            
            self._data = [json.loads(l) for l in f.readlines()]

        with open(
            self.path.joinpath( f"tables.json"),
            "r",
            encoding="utf8",
        ) as f:
            self.tables = json.load(f)
        self.tables = {table["db_id"]: table for table in self.tables}
        self.preprocess_steps =[]
        self.postprocess_steps = []

    def __getitem__(self, idx: int) -> BirdCriticDatasetItem:
        return BirdCriticDatasetItem(
            gt=''.join(x for x in self._data[idx]["sol_sql"]),
            query=''.join(x for x in self._data[idx]["issue_sql"]),
            preprocess_steps=''.join(x for x in self._data[idx]["preprocess_sql"]),
            postprocess_steps=''.join(x for x in self._data[idx]["clean_up_sql"]),
            info=self._data[idx],
            path=self.path
        )

    def __len__(self) -> int:
        return len(self._data)