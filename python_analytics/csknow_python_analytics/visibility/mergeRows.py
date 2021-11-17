import argparse
import math

import pytesseract
import cv2
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import os
import re
import psycopg2
from dataclasses import dataclass
import time

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="input file to merge rows in",
                    type=str)
parser.add_argument("output_dir", help="output directory to write rows",
                    type=str)
args = parser.parse_args()
df_input = pd.read_csv(args.input_file, index_col='id')
df_sorted = df_input.sort_values(['demo', 'spotter', 'spotted', 'start_game_tick'])
dicts_sorted = df_sorted.to_dict(orient='records')
dicts_output = []
i = 0
while i < len(dicts_sorted):
    j = 1
    while j < len(dicts_sorted) - i:
        next_row = dicts_sorted[i+j]
        a = dicts_sorted[i]['demo'] != next_row['demo']
        b = dicts_sorted[i]['spotter'] != next_row['spotter']
        c = dicts_sorted[i]['spotted'] != next_row['spotted']
        d = dicts_sorted[i]['end_game_tick'] + 2 < next_row['start_game_tick']
        if dicts_sorted[i]['demo'] != next_row['demo'] or \
            dicts_sorted[i]['spotter'] != next_row['spotter'] or \
            dicts_sorted[i]['spotted'] != next_row['spotted'] or \
            dicts_sorted[i]['end_game_tick'] + 2 < next_row['start_game_tick']:
            break
        else:
            dicts_sorted[i]['end_game_tick'] = next_row['end_game_tick']
            j += 1
    dicts_output.append(dicts_sorted[i])
    i += j

df_output_unsorted = pd.DataFrame(dicts_output)
df_output = df_output_unsorted.sort_values(['demo', 'start_game_tick'])
df_output.to_csv(args.output_dir + "/" + os.path.basename(args.input_file), index_label='id')
