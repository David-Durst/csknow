from pathlib import Path
from typing import Dict

rank_map = {
    'NA': 0,
    'S1': 1,
    'S2': 2,
    'S3': 3,
    'S4': 4,
    'SE': 5,
    'SEM': 6,
    'GN1': 7,
    'GN2': 8,
    'GN3': 9,
    'GNM': 10,
    'MG1': 11,
    'MG2': 12,
    'MGE': 13,
    'DMG': 14,
    'LE': 15,
    'LEM': 16,
    'SMFC': 17,
    'GE': 18,
    'Global': 18
}

rank_reverse_map: Dict[int, str] = {}
for k, v in rank_map.items():
    if v not in rank_reverse_map:
        rank_reverse_map[v] = k
    rank_reverse_map[0] = 'None'

bot_types = ['human', 'learned', 'hand-crafted', 'default']
nice_bot_types = ['Human', 'CSMoveBot', 'ManualBot', 'CSGOBot']
answer_key = [
    ['human', 'default', 'learned', 'hand-crafted'],
    ['hand-crafted', 'human', 'learned', 'default'],
    ['learned', 'human', 'default', 'hand-crafted'],
    ['default', 'hand-crafted', 'learned', 'human'],
    ['human', 'default', 'learned', 'hand-crafted'],
    ['default', 'hand-crafted', 'human', 'learned'],
    ['learned', 'human', 'hand-crafted', 'default'],
    ['hand-crafted', 'learned', 'human', 'default']
]
file_name_col = 'File'
rank_col = 'Rank'
years_exp_col = 'Years of Development Experience'
example_col = 'Example'
high_skill_col = 'High Skill'
user_study_plots = Path(__file__).parent / 'user_plots'
