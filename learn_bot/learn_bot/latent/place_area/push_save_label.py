from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Dict, Optional


class PushSaveLabel(IntEnum):
    FullPush = 1
    Partial = 2
    FullSave = 3


@dataclass
class PushSaveRoundData:
    label: PushSaveLabel
    percent_push: float
    demo_file: str
    round_id: int
    round_number: int


label_str = 'label'
percent_push_str = 'percent push'
demo_file_str = 'demo file'
round_id_str = 'round id'
round_number_str = 'round number'


class PushSaveRoundLabels:
    round_id_to_data: Dict[int, PushSaveRoundData]

    def __init__(self, p: Optional[Path] = None):
        if p is None:
            self.round_id_to_data = {}
        else:
            self.load(p)

    def load(self, p: Path):
        with open(p, 'r') as f:
            unparsed_dict: Dict[int, Dict] = eval(f.read())
        self.round_id_to_data = {
            k: PushSaveRoundData(PushSaveLabel(v[label_str]), v[percent_push_str], v[demo_file_str],
                                 v[round_id_str], v[round_number_str])
            for k, v in unparsed_dict
        }

    def save(self, p:Path):
        with open(p, 'w') as f:
            f.write(str(self.to_pod_dict()))

    def to_pod_dict(self) -> Dict[int, Dict]:
        return {k: {
            label_str: int(v.label),
            percent_push_str: v.percent_push,
            demo_file_str: v.demo_file,
            round_id_str: v.round_id,
            round_number_str: v.round_number
        } for k, v in self.round_id_to_data.items()}

