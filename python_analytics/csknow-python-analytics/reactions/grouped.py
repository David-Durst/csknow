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
    'avg(hand_aim_react_s) as avg_aim_hand_react, avg(cpu_aim_react_s) as avg_aim_cpu_react, ' + \
    'sum(case when abs(hand_aim_react_s) <= 0.2 then 1 else 0 end) as hand_preaims, ' + \
    'sum(case when abs(cpu_aim_react_s) <= 0.2 then 1 else 0 end) as cpu_preaims'
group_cols = f'''group by game_id, round_id / {args.grouping_rounds}, spotter_id, spotter, hacking, distinct_others_spotted_during_time'''
hand_filtered_df = sqlio.read_sql_query(f'''select {select_cols} from {hand_react_reasonable} {group_cols}''', conn)
cpu_filtered_df = sqlio.read_sql_query(f'''select {select_cols} from {cpu_react_reasonable} {group_cols}''', conn)

hacks_hand_filtered_df = hand_filtered_df[hand_filtered_df['hacking']]
hacks_cpu_filtered_df = cpu_filtered_df[cpu_filtered_df['hacking']]
legit_hand_filtered_df = hand_filtered_df[~hand_filtered_df['hacking']]
legit_cpu_filtered_df = cpu_filtered_df[~cpu_filtered_df['hacking']]

print(f'''hacks hand size {len(hacks_hand_filtered_df)}, legit hand size {len(legit_hand_filtered_df)} \n ''' +
      f'''hacks cpu size {len(hacks_cpu_filtered_df)}, legit cpu size {len(legit_cpu_filtered_df)}''')


def makePlotterFunction(bin_width, pct):
    def plotPctWith200MSBins(df, col, ax):
        num_bins = math.ceil((df[col].max() - df[col].min()) / bin_width)
        df.hist(col, bins=num_bins, ax=ax, weights=np.ones(len(df[col])) / len(df[col]))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        return pd.cut(df[col], num_bins).value_counts().sort_index()

    def plotNumWith200MSBins(df, col, ax):
        num_bins = math.ceil((df[col].max() - df[col].min()) / bin_width)
        df.hist(col, bins=num_bins, ax=ax)
        return pd.cut(df[col], num_bins).value_counts().sort_index()

    if pct:
        return plotPctWith200MSBins
    else:
        return plotNumWith200MSBins


def makeHistograms(hand_col, cpu_col, plotting_function, change_x_lim, change_y_lim, name, x_label):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
    hacks_hand_distribution = plotting_function(hacks_hand_filtered_df, hand_col, ax[0][0])
    legit_hand_distribution = plotting_function(legit_hand_filtered_df, hand_col, ax[0][1])
    hacks_cpu_distribution = plotting_function(hacks_cpu_filtered_df, cpu_col, ax[1][0])
    legit_cpu_distribution = plotting_function(legit_cpu_filtered_df, cpu_col, ax[1][1])

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            if change_x_lim:
                ax[i][j].set_xlim(-3, 3)
            if change_y_lim:
                if i == 0:
                    ax[i][j].set_ylim(0, max(hacks_hand_distribution.max(), legit_hand_distribution.max()) + 2)
                else:
                    ax[i][j].set_ylim(0, max(hacks_cpu_distribution.max(), legit_cpu_distribution.max()) + 1)
            ax[i][j].set_xlabel(x_label, fontsize=14)
            ax[i][j].set_ylabel('Frequency', fontsize=14)

    ax[0][0].set_title('Hand Labeled, Hacking', fontsize=18)
    ax[0][1].set_title('Hand Labeled, Not Hacking', fontsize=18)
    ax[1][0].set_title('CPU Labeled, Hacking', fontsize=18)
    ax[1][1].set_title('CPU Labeled, Not Hacking', fontsize=18)

    ax[0][0].annotate('total points: ' + str(len(hacks_hand_filtered_df)), (0,2), fontsize="14")
    ax[0][1].annotate('total points: ' + str(len(legit_hand_filtered_df)), (0,2), fontsize="14")
    ax[1][0].annotate('total points: ' + str(len(hacks_cpu_filtered_df)), (0,2), fontsize="14")
    ax[1][1].annotate('total points: ' + str(len(legit_cpu_filtered_df)), (0,2), fontsize="14")

    plt.tight_layout()
    fig.savefig(args.plot_folder + name + '_grouped_histogram__hand_vs_cpu__hacking_vs_legit.png')

    plt.clf()


makeHistograms('avg_aim_hand_react', 'avg_aim_cpu_react', makePlotterFunction(0.2, False), True, True, 'count_avg', 'Reaction Time (s)')
makeHistograms('avg_aim_hand_react', 'avg_aim_cpu_react', makePlotterFunction(0.2, True), True, False, 'pct_avg', 'Reaction Time (s)')
makeHistograms('hand_preaims', 'cpu_preaims', makePlotterFunction(1, False), False, True, 'count_preaim', 'Number of Preaims')
makeHistograms('hand_preaims', 'cpu_preaims', makePlotterFunction(1, True), False, False, 'pct_preaim', 'Number of Preaims')

def makeLogReg(df, col, name):
    plt.clf()
    X_df = df[[col, 'distinct_others_spotted_during_time']]
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

makeLogReg(hand_filtered_df, 'avg_aim_hand_react', 'Hand')
makeLogReg(cpu_filtered_df, 'avg_aim_cpu_react', 'CPU')
