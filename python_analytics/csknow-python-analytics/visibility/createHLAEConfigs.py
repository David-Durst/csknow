import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument("entities_file", help="file with output of mirv_listEntities isPlayer=1 for demo",
                    type=str)
args = parser.parse_args()

with open(args.entities_file) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

num_entities = len(lines) // 5

for entity_index in range(num_entities):
    entity_index_and_handle_line = lines[entity_index * num_entities]
    player_name_line = lines[entity_index * num_entities + 2]
