import re
import string

import psycopg2
import argparse
import pandas as pd
import pandas.io.sql as sqlio
import pathlib
from dataclasses import dataclass
import os


parser = argparse.ArgumentParser()
parser.add_argument("entities_file", help="file with output of mirv_listEntities isPlayer=1 for demo",
                    type=str)
parser.add_argument("compute_visibility", help="path to computeVisibilityGenerated.sh folder",
                    type=str)
parser.add_argument("video_dir", help="path to video directory",
                    type=str)
parser.add_argument("password", help="database password",
                    type=str)
parser.add_argument("hacking", help="1 if hacking, 0 if not",
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
                # skip coaches, other people with steam id not in match
                if len(players_and_teams.loc[players_and_teams['name'] == match_line_1.group(1)]) == 0:
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
colors_for_analysis = ['red', 'redGreen', 'green', 'greenBlue', 'blue', 'redBlue']
actions_to_analysis = { 'afxWhTRed':'red', 'afxWhTGreen':'green', 'afxWhTBlue':'blue',
                        'afxWhTRedBlue':'redBlue', 'afxWhTGreenBlue':'greenBlue'}
enemy_to_color_per_team = {}
for team in teams:
    enemy_to_color_per_team[team] = {}
    for color in colors_for_analysis:
        enemy_to_color_per_team[team][color] = ""

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
                enemy_to_color_per_team[team][actions_to_analysis[color_actions[color_counter]]] = p.name
                color_counter += 1
        team_f.write(f'''demo_goto {game_tick_for_getting_teams}\n''')

video_path = os.path.join(args.video_dir, prefix)
if not os.path.exists(video_path):
    os.mkdir(video_path)
player_config_file_names = []
compute_visibility_sh = open(os.path.join(args.compute_visibility, 'computeVisibilityGenerated.sh'), 'a')
compute_visibility_sh.write(f'''\n#{prefix}\n''')
for player in players:
    if player.xuid == '0':
        continue
    clean_name = re.sub(r'\W+', '', player.name)
    player_config_file_name = prefix + '_pre_load_' + clean_name + '_' + player.team + '.cfg'
    player_file_path = file_dir / player_config_file_name
    player_config_file_names.append(player_config_file_name)
    with open(player_file_path, 'w+') as player_f:
        player_f.write(f'''mirv_pov {player.id}\n''')
        player_f.write(f'''playdemo {demo_name}\n''')
    players_by_color = []
    for color in colors_for_analysis:
        players_by_color.append(enemy_to_color_per_team[player.team][color])
    compute_visibility_sh.write(f'''python computeVisibility.py ${{script_dir}}/videos/{prefix}/{prefix}_{clean_name}_{player.team}.mp4 ${{script_dir}}/../local_data/visibilities/ ${{script_dir}}/visibilityLogs/ "{player.name}" "{','.join(players_by_color)}" {args.hacking} {demo_name} ${{pass}}\n''')
compute_visibility_sh.write(f'''\n\n''')
compute_visibility_sh.close()

record_gameplay_bat = open(os.path.join(args.compute_visibility, 'recordGameplay.bat'), 'a')
record_gameplay_bat.write(f'''\n:: #{prefix}\n''')
record_gameplay_bat.write('python recordGameplay.py %script_dir%visibilitySignImages\\tick0.png C:\\Users\\Administrator\\Videos\\ ')
record_gameplay_bat.write(",".join(player_config_file_names))
record_gameplay_bat.write('\n')

record_player_state_bat = open(os.path.join(args.compute_visibility, 'recordPlayerState.bat'), 'a')
for config_file_name in player_config_file_names:
    record_player_state_bat.write(f'''\n:: #{prefix}\n''')
    record_player_state_bat.write(':: python recordPLayerState.py %script_dir%visibilitySignImages\\just_death.png %script_dir%visibilitySignImages\\tick_no_death.png C:\\Users\\Administrator\\Documents\\ ')
    record_player_state_bat.write(config_file_name)
    record_player_state_bat.write('\n')
