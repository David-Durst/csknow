import psycopg2
import argparse
import pandas.io.sql as sqlio
from dataclasses import dataclass
from plottingHelpers import *
from regressionHelpers import *

parser = argparse.ArgumentParser()
parser.add_argument("password", help="database password",
                    type=str)
parser.add_argument("query_file", help="file containing query",
                    type=str)
parser.add_argument("plot_folder", help="folder for plots",
                    type=str)
parser.add_argument("grouping_rounds", help="number of rounds to group together",
                    type=int)
args = parser.parse_args()
print(args)

conn = psycopg2.connect(
    host="localhost",
    database="csknow",
    user="postgres",
    password=args.password,
    port=3125)
cur = conn.cursor()

with open(args.query_file, 'r') as query_file:
    cur.execute(query_file.read())

hand_react_reasonable = '(select * from react_final where abs(hand_aim_react_s) <= 3) as hand_react_reasonable'
cpu_react_reasonable = '(select * from react_final where abs(cpu_aim_react_s) <= 3) as cpu_react_reasonable'
select_cols = 'game_id, count(*) as num, min(round_id) as min_round_id, max(round_id) as max_round_id, spotter_id, spotter, hacking, ' + \
    'avg(distinct_others_spotted_during_time) as distinct_others_spotted_during_time, ' + \
    'avg(coalesce(hand_aim_react_s, 6.0)) as avg_aim_hand_react, avg(coalesce(cpu_aim_react_s, 4.0)) as avg_aim_cpu_react, ' + \
    'avg(coalesce(hand_fire_react_s, 6.0)) as avg_fire_hand_react, avg(coalesce(cpu_fire_react_s, 4.0)) as avg_fire_cpu_react, ' + \
    'sum(case when hand_aim_react_s < -1.5 then 1 else 0 end) as hand_preaims, ' + \
    'sum(case when cpu_aim_react_s < -1.5 then 1 else 0 end) as cpu_preaims'
group_cols = f'''group by game_id, round_id / {args.grouping_rounds}, spotter_id, spotter, hacking'''
order_cols = f'''order by game_id, min_round_id, spotter'''
hand_filtered_df = sqlio.read_sql_query(f'''select {select_cols} from {hand_react_reasonable} {group_cols} {order_cols}''', conn)
cpu_filtered_df = sqlio.read_sql_query(f'''select {select_cols} from {cpu_react_reasonable} {group_cols} {order_cols}''', conn)


@dataclass(frozen=True)
class LabeledData:
    pro_hand_filtered_df: pd.DataFrame
    pro_cpu_filtered_df: pd.DataFrame
    hacks_hand_filtered_df: pd.DataFrame
    hacks_cpu_filtered_df: pd.DataFrame
    legit_hand_filtered_df: pd.DataFrame
    legit_cpu_filtered_df: pd.DataFrame

    def get_as_grid(self):
        return [[self.pro_hand_filtered_df, self.hacks_hand_filtered_df, self.legit_hand_filtered_df],
                [self.pro_cpu_filtered_df, self.hacks_cpu_filtered_df, self.legit_cpu_filtered_df]]

pro_hand_filtered_df = hand_filtered_df[hand_filtered_df['hacking'] == 2]
pro_cpu_filtered_df = cpu_filtered_df[cpu_filtered_df['hacking'] == 2]
hacks_hand_filtered_df = hand_filtered_df[hand_filtered_df['hacking'] == 1]
hacks_cpu_filtered_df = cpu_filtered_df[cpu_filtered_df['hacking'] == 1]
legit_hand_filtered_df = hand_filtered_df[hand_filtered_df['hacking'] == 0]
legit_cpu_filtered_df = cpu_filtered_df[cpu_filtered_df['hacking'] == 0]

dfs = LabeledData(pro_hand_filtered_df, pro_cpu_filtered_df, hacks_hand_filtered_df, hacks_cpu_filtered_df, legit_hand_filtered_df, legit_cpu_filtered_df)


print(f'''pro hand size {len(pro_hand_filtered_df)}, hacks hand size {len(hacks_hand_filtered_df)}, legit hand size {len(legit_hand_filtered_df)} \n ''' +
      f'''pro cpu size {len(pro_cpu_filtered_df)}, hacks cpu size {len(hacks_cpu_filtered_df)}, legit cpu size {len(legit_cpu_filtered_df)}''')

plot_titles = [['Pixel Labeled, Pro', 'Pixel Labeled, Hacking', 'Pixel Labeled, Not Hacking'], ['BBox Labeled, Pro', 'BBox Labeled, Hacking', 'BBox Labeled, Not Hacking']]

makeHistograms(dfs.get_as_grid(), ['avg_aim_hand_react', 'avg_aim_cpu_react'],
               makePlotterFunction(0.2, False), plot_titles, 'Grouped Count Aim Reactions', 'Aim Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), ['avg_aim_hand_react', 'avg_aim_cpu_react'],
               makePlotterFunction(0.2, True), plot_titles, 'Grouped Percent Aim Reactions', 'Aim Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), ['avg_fire_hand_react', 'avg_fire_cpu_react'],
               makePlotterFunction(0.5, False), plot_titles, 'Grouped Count Fire Reactions', 'Fire Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), ['avg_fire_hand_react', 'avg_fire_cpu_react'],
               makePlotterFunction(0.5, True), plot_titles, 'Grouped Percent Fire Reactions', 'Fire Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), ['hand_preaims', 'cpu_preaims'],
               makePlotterFunction(1, False), plot_titles, 'Grouped Count Pre-Aims', 'Number of Pre-Aims', args.plot_folder)
makeHistograms(dfs.get_as_grid(), ['hand_preaims', 'cpu_preaims'],
               makePlotterFunction(1, True), plot_titles, 'Grouped Percent Pre-Aims', 'Number of Pre-Aims', args.plot_folder)


hand_just_legit_cheat_df = hand_filtered_df[hand_filtered_df['hacking'] < 2]
cpu_just_legit_cheat_df = cpu_filtered_df[cpu_filtered_df['hacking'] < 2]
#makeLogReg(hand_filtered_df, ['avg_aim_hand_react'], 'GPU')
makeLogReg(hand_just_legit_cheat_df, ['avg_aim_hand_react', 'avg_fire_hand_react', 'hand_preaims'], 'Pixel')
#makeLogReg(cpu_filtered_df, ['avg_aim_cpu_react'], 'CPU')
makeLogReg(cpu_just_legit_cheat_df, ['avg_aim_cpu_react', 'avg_fire_cpu_react', 'cpu_preaims'], 'BBox')
