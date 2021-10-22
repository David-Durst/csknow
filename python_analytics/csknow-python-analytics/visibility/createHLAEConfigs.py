import re

import psycopg2
import argparse
import pandas as pd
import pandas.io.sql as sqlio
import pathlib
from dataclasses import dataclass


parser = argparse.ArgumentParser()
parser.add_argument("entities_file", help="file with output of mirv_listEntities isPlayer=1 for demo",
                    type=str)
parser.add_argument("password", help="database password",
                    type=str)
args = parser.parse_args()

conn = psycopg2.connect(
    host="localhost",
    database="csknow",
    user="postgres",
    password=args.password,
    port=3125)

@dataclass(frozen=True)
class PlayerEntity:
    name: str
    xuid: str
    id: str
    handle: str
    team: str

players = []
teams = set([])

file_dir = pathlib.Path(args.entities_file).parent

with open(args.entities_file) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

entities_offset = 2
lines_per_entity = 5
num_entities = (len(lines) - 1) // lines_per_entity

demo_name = lines[0]
game_tick_for_getting_teams = lines[1]

players_and_teams = sqlio.read_sql_query(f'''
    select p.name as name, p.id, pat.team as team
    from games g
    join players p on g.id = p.game_id
    join player_at_tick pat on p.id = pat.player_id
    join ticks t on t.id = pat.tick_id
    where g.demo_file = '{demo_name}'
    and t.game_tick_number = {game_tick_for_getting_teams}''', conn)

for entity_index in range(num_entities):
    entity_index_and_handle_line = lines[entity_index * lines_per_entity + entities_offset]
    player_xuid_line = lines[entity_index * lines_per_entity + 1 + entities_offset]
    player_name_line = lines[entity_index * lines_per_entity + 2 + entities_offset]
    if match_line_0 := re.search('(\d+) .*Player:: :(\d+)', entity_index_and_handle_line):
        if match_line_1 := re.search('playerInfo.name: (.+)', player_name_line):
            if match_line_2 := re.search('playerInfo.xuid: (.+)', player_xuid_line):
                # skip gotv and bots as not using their perspective
                if match_line_1.group(1) == 'GOTV':
                    continue
                team = str(players_and_teams.loc[players_and_teams['name'] == match_line_1.group(1), 'team'].item())
                players.append(PlayerEntity(match_line_1.group(1), match_line_2.group(1), match_line_0.group(1), match_line_0.group(2), team))
                teams.add(team)
            else:
                print(f'''missed on xuid line for entity {entity_index}''')
        else:
            print(f'''missed on name line for entity {entity_index}''')
            exit(1)
    else:
        print(f'''missed on entity index and handle line for entity {entity_index}''')
        exit(1)

input_name = pathlib.Path(args.entities_file).with_suffix('').stem
if match_prefix := re.search('(.+)_entities', input_name):
    prefix = match_prefix.group(1)
else:
    print(f'''Invalid entities file name {args.entities_file}, missing _entities at end before .dem''')
    exit(1)

color_actions = ['afxWhTRed', 'afxWhTGreen', 'afxWhTBlue', 'afxWhTRedBlue', 'afxWhTGreenBlue']
for team in teams:
    color_counter = 0
    team_file_path = file_dir / (prefix + '_post_load_' + team + '.cfg')
    with open(team_file_path, 'w+') as team_f:
        team_f.write('exec create_bfs_stream\n')
        for p in players:
            if p.team == team:
                team_f.write(f'''mirv_streams edit bfs actionFilter addEx "handle={p.handle}" "action=noDraw" // {p.name}\n''')
            else:
                team_f.write(f'''mirv_streams edit bfs actionFilter addEx "handle={p.handle}" "action={color_actions[color_counter]}" // {p.name}\n''')
                color_counter += 1

for player in players:
    if player.xuid == '0':
        continue
    player_file_path = file_dir / (prefix + '_pre_load_' + player.name[:5].rstrip() + '_' + player.team + '.cfg')
    with open(player_file_path, 'w+') as player_f:
        player_f.write(f'''mirv_pov {p.id}\n''')
        player_f.write(f'''playdemo {demo_name}\n''')
        player_f.write(f'''demo_goto {game_tick_for_getting_teams}\n''')
