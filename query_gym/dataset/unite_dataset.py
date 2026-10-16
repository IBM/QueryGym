import json
from pathlib import Path
from typing import Dict, List

from sqlgym.datasets.utils import DbDataset, DbDatasetItem

        
class UniteDataset(DbDataset):
    def __init__(self, dataset_path, mode):
        self.path = Path(dataset_path)
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

    def __getitem__(self, idx: int) -> DbDatasetItem:
        return DbDatasetItem(
            path="file:"
            + self.path.joinpath(
                "database",
                self._data[idx]["db_id"],
                f'{self._data[idx]["db_id"]}.sqlite',
            ).as_posix()
            + "?mode=ro",
            query=self._data[idx]['question'],
            gt=self._data[idx]["query"],
            info=self._data[idx]
        )

    def __len__(self) -> int:
        return len(self._data)