import psycopg2
import argparse
import pandas.io.sql as sqlio
from dataclasses import dataclass
from plottingHelpers import *
from regressionHelpers import *
from dataframeHelpers import *

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

unfiltered_table = 'react_final'
filtered_table = '(select * from react_final where abs(aim_react_s) <= 3 and not seen_last_five_seconds) filtered_table'
select_cols = 'game_id, visibility_technique_id, count(*) as num, min(round_id) as min_round_id, max(round_id) as max_round_id, spotter_id, spotter, hacking, ' + \
    'avg(distinct_others_spotted_during_time) as distinct_others_spotted_during_time, ' + \
    'avg(aim_react_s) as avg_aim_react_s, avg(case when fire_react_s is not null and fire_react_s <= 2.0 then fire_react_s else NULL end) as avg_fire_react_s, ' + \
    'sum(case when aim_react_s < -1.5 then 1 else 0 end) as preaims, ' + \
    'avg(clusters_covered) as avg_pct_clusters_covered'
group_cols = f'''group by hacking, visibility_technique_id, game_id, round_id / {args.grouping_rounds}, spotter_id, spotter'''
order_cols = f'''order by game_id, min_round_id, spotter'''
unfiltered_df = sqlio.read_sql_query(f'''select {select_cols} from {unfiltered_table} {group_cols} {order_cols}''', conn)
filtered_df = sqlio.read_sql_query(f'''select {select_cols} from {filtered_table} {group_cols} {order_cols}''', conn).dropna(subset=['avg_aim_react_s', 'avg_fire_react_s'])


dfs = VisibilityTechniqueDataFrames(unfiltered_df, filtered_df)
dfs.print_size()


makeHistograms(dfs.get_as_grid(), 'avg_aim_react_s', makePlotterFunction(0.2, False), plot_titles,
               'Grouped Count Aim Reactions', 'Aim Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'avg_aim_react_s', makePlotterFunction(0.2, True), plot_titles,
               'Grouped Percent Aim Reactions', 'Aim Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'avg_fire_react_s', makePlotterFunction(0.5, False),
               plot_titles, 'Grouped Count Fire Reactions', 'Fire Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'avg_fire_react_s', makePlotterFunction(0.5, True),
               plot_titles, 'Grouped Percent Fire Reactions', 'Fire Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'preaims', makePlotterFunction(1, False),
               plot_titles, 'Grouped Count Pre-Aims', 'Number of Pre-Aims', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'preaims', makePlotterFunction(1, True),
               plot_titles, 'Grouped Percent Pre-Aims', 'Number of Pre-Aims', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'avg_pct_clusters_covered', makePlotterFunction(0.1, False, 1.0, 0.0),
               plot_titles, 'Grouped Count Avg Pct Clusters Covered', 'Pct Clusters Covered', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'avg_pct_clusters_covered', makePlotterFunction(0.1, True, 1.0, 0.0),
               plot_titles, 'Grouped Percent Avg Pct Clusters Covered', 'Pct Clusters Covered', args.plot_folder)

input_cols = ['avg_aim_react_s', 'avg_fire_react_s', 'preaims', 'avg_pct_clusters_covered']
grouped_str = 'grouped'
makeLogReg(dfs.pix_adjusted_dfs.get_hacks_union_legit(), input_cols, visibility_techniques[0], grouped_str, args.plot_folder)
makeLogReg(dfs.pix_unadjusted_dfs.get_hacks_union_legit(), input_cols, visibility_techniques[1], grouped_str, args.plot_folder)
makeLogReg(dfs.bbox_dfs.get_hacks_union_legit(), input_cols, visibility_techniques[2], grouped_str, args.plot_folder)
#hand_just_legit_cheat_df = hand_filtered_df[hand_filtered_df['hacking'] < 2]
#cpu_just_legit_cheat_df = cpu_filtered_df[cpu_filtered_df['hacking'] < 2]
##makeLogReg(hand_filtered_df, ['avg_aim_hand_react'], 'GPU')
#makeLogReg(hand_just_legit_cheat_df, ['avg_aim_hand_react', 'avg_fire_hand_react', 'hand_preaims'], 'Pixel')
##makeLogReg(cpu_filtered_df, ['avg_aim_cpu_react'], 'CPU')
#makeLogReg(cpu_just_legit_cheat_df, ['avg_aim_cpu_react', 'avg_fire_cpu_react', 'cpu_preaims'], 'BBox')
