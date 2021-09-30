import math

import psycopg2
import argparse
import pandas as pd
import pandas.io.sql as sqlio
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import numpy as np
from dataclasses import dataclass

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
select_cols = 'game_id, min(round_id) as min_round_id, max(round_id) as max_round_id, spotter_id, spotter, hacking, ' + \
    'distinct_others_spotted_during_time, ' + \
    'avg(coalesce(hand_aim_react_s, 4.0)) as avg_aim_hand_react, avg(coalesce(cpu_aim_react_s, 4.0)) as avg_aim_cpu_react, ' + \
    'avg(coalesce(hand_fire_react_s, 4.0)) as avg_fire_hand_react, avg(coalesce(cpu_fire_react_s, 4.0)) as avg_fire_cpu_react, ' + \
    'sum(case when hand_aim_react_s < 0.15 then 1 else 0 end) as hand_preaims, ' + \
    'sum(case when cpu_aim_react_s < 0.15 then 1 else 0 end) as cpu_preaims'
group_cols = f'''group by game_id, round_id / {args.grouping_rounds}, spotter_id, spotter, hacking, distinct_others_spotted_during_time'''
hand_filtered_df = sqlio.read_sql_query(f'''select {select_cols} from {hand_react_reasonable} {group_cols}''', conn)
cpu_filtered_df = sqlio.read_sql_query(f'''select {select_cols} from {cpu_react_reasonable} {group_cols}''', conn)


@dataclass(frozen=True)
class LabeledData:
    hacks_hand_filtered_df: pd.DataFrame
    hacks_cpu_filtered_df: pd.DataFrame
    legit_hand_filtered_df: pd.DataFrame
    legit_cpu_filtered_df: pd.DataFrame

hacks_hand_filtered_df = hand_filtered_df[hand_filtered_df['hacking']]
hacks_cpu_filtered_df = cpu_filtered_df[cpu_filtered_df['hacking']]
legit_hand_filtered_df = hand_filtered_df[~hand_filtered_df['hacking']]
legit_cpu_filtered_df = cpu_filtered_df[~cpu_filtered_df['hacking']]

dfs = LabeledData(hacks_hand_filtered_df, hacks_cpu_filtered_df, legit_hand_filtered_df, legit_cpu_filtered_df)


print(f'''hacks hand size {len(hacks_hand_filtered_df)}, legit hand size {len(legit_hand_filtered_df)} \n ''' +
      f'''hacks cpu size {len(hacks_cpu_filtered_df)}, legit cpu size {len(legit_cpu_filtered_df)}''')


def makePlotterFunction(bin_width, pct):
    def plotPctWith200MSBins(df, col, ax):
        col_vals = df[col].dropna()
        num_bins = math.ceil((col_vals.max() - col_vals.min()) / bin_width) + 1
        df.hist(col, bins=num_bins, ax=ax, weights=np.ones(len(col_vals)) / len(col_vals))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        return pd.cut(col_vals, num_bins).value_counts(normalize=True).sort_index()

    def plotNumWith200MSBins(df, col, ax):
        col_vals = df[col].dropna()
        num_bins = math.ceil((col_vals.max() - col_vals.min()) / bin_width) + 1
        df.hist(col, bins=num_bins, ax=ax)
        return pd.cut(col_vals, num_bins).value_counts().sort_index()

    if pct:
        return plotPctWith200MSBins
    else:
        return plotNumWith200MSBins


def makeHistograms(dfs, hand_col, cpu_col, plotting_function, name, x_label):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
    hacks_hand_distribution = plotting_function(dfs.hacks_hand_filtered_df, hand_col, ax[0][0])
    legit_hand_distribution = plotting_function(dfs.legit_hand_filtered_df, hand_col, ax[0][1])
    hacks_cpu_distribution = plotting_function(dfs.hacks_cpu_filtered_df, cpu_col, ax[1][0])
    legit_cpu_distribution = plotting_function(dfs.legit_cpu_filtered_df, cpu_col, ax[1][1])

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            (x0_min, x0_max) = ax[i][0].get_xlim()
            (x1_min, x1_max) = ax[i][1].get_xlim()
            ax[i][j].set_xlim(min(x0_min, x1_min), min(x0_max, x1_max))
            if i == 0:
                ax[i][j].set_ylim(0, max(hacks_hand_distribution.max(), legit_hand_distribution.max()) * 1.1)
            else:
                ax[i][j].set_ylim(0, max(hacks_cpu_distribution.max(), legit_cpu_distribution.max()) * 1.1)
            ax[i][j].set_xlabel(x_label, fontsize=14)
            ax[i][j].set_ylabel('Frequency', fontsize=14)

    ax[0][0].set_title('Hand Labeled, Hacking', fontsize=18)
    ax[0][1].set_title('Hand Labeled, Not Hacking', fontsize=18)
    ax[1][0].set_title('CPU Labeled, Hacking', fontsize=18)
    ax[1][1].set_title('CPU Labeled, Not Hacking', fontsize=18)

    def get_num_points_coordinate(ax, x_pct = 0.6, y_pct = 0.8):
        (y_min, y_max) = ax.get_ylim()
        (x_min, x_max) = ax.get_xlim()
        return (x_min + (x_max-x_min) * x_pct, y_min + (y_max-y_min)*y_pct)

    ax[0][0].annotate('total points: ' + str(len(hacks_hand_filtered_df[hand_col].dropna())), get_num_points_coordinate(ax[0][0]), fontsize="14")
    ax[0][1].annotate('total points: ' + str(len(legit_hand_filtered_df[hand_col].dropna())), get_num_points_coordinate(ax[0][1]), fontsize="14")
    ax[1][0].annotate('total points: ' + str(len(hacks_cpu_filtered_df[cpu_col].dropna())), get_num_points_coordinate(ax[1][0]), fontsize="14")
    ax[1][1].annotate('total points: ' + str(len(legit_cpu_filtered_df[cpu_col].dropna())), get_num_points_coordinate(ax[1][1]), fontsize="14")

    plt.tight_layout()
    fig.savefig(args.plot_folder + name + '_grouped_histogram__hand_vs_cpu__hacking_vs_legit.png')

    plt.clf()

makeHistograms(dfs, 'avg_aim_hand_react', 'avg_aim_cpu_react', makePlotterFunction(0.2, False), 'count_aim', 'Aim Reaction Time (s)')
makeHistograms(dfs, 'avg_aim_hand_react', 'avg_aim_cpu_react', makePlotterFunction(0.2, True), 'pct_aim', 'Aim Reaction Time (s)')
makeHistograms(dfs, 'avg_fire_hand_react', 'avg_fire_cpu_react', makePlotterFunction(0.5, False), 'count_fire', 'Fire Reaction Time (s)')
makeHistograms(dfs, 'avg_fire_hand_react', 'avg_fire_cpu_react', makePlotterFunction(0.5, True), 'pct_fire', 'Fire Reaction Time (s)')
makeHistograms(dfs, 'hand_preaims', 'cpu_preaims', makePlotterFunction(1, False), 'count_preaim', 'Number of Preaims')
makeHistograms(dfs, 'hand_preaims', 'cpu_preaims', makePlotterFunction(1, True), 'pct_preaim', 'Number of Preaims')

def makeLogReg(df, cols, name):
    plt.clf()
    X_df = df[cols + ['distinct_others_spotted_during_time']]
    y_series = df['hacking']

    X_train, X_test, y_train, y_test = train_test_split(X_df,y_series,test_size=0.2,random_state=42)

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.set(font_scale=1.5)
    confusion_matrix_heatmap = sn.heatmap(confusion_matrix, annot=True)
    plt.title(name + ' Labeled Confusion Matrix', fontsize=24)
    confusion_matrix_figure = confusion_matrix_heatmap.get_figure()
    confusion_matrix_figure.savefig(args.plot_folder + name + '_grouped_confusion_matrix__hand_vs_cpu__hacking_vs_legit.png')

    print(name + ' coeff: ', lr_model.coef_)
    print(name + ' accuracy: ', metrics.accuracy_score(y_test, y_pred))
    #print(np.argwhere(y_pred == False))

#makeLogReg(hand_filtered_df, ['avg_aim_hand_react'], 'Hand')
makeLogReg(hand_filtered_df, ['avg_aim_hand_react', 'hand_preaims'], 'Hand')
#makeLogReg(cpu_filtered_df, ['avg_aim_cpu_react'], 'CPU')
makeLogReg(cpu_filtered_df, ['avg_aim_cpu_react', 'cpu_preaims'], 'CPU')
