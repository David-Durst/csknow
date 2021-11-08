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
parser.add_argument("plot_folder", help="folder_for_plots",
                    type=str)
parser.add_argument("visibility_technique_id", help="folder_for_plots",
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

unfiltered_df = sqlio.read_sql_query("select * from react_final", conn)
hand_filtered_df = sqlio.read_sql_query("select * from react_final where aim_react_s <= 3 and aim_react_s >= -3", conn)
cpu_filtered_df = sqlio.read_sql_query("select * from react_final where aim_react_s <= 3 and aim_react_s >= -3", conn)

hacks_hand_filtered_df = hand_filtered_df[hand_filtered_df['hacking'] == 1]
hacks_cpu_filtered_df = cpu_filtered_df[cpu_filtered_df['hacking'] == 1]
legit_hand_filtered_df = hand_filtered_df[hand_filtered_df['hacking'] == 0]
legit_cpu_filtered_df = cpu_filtered_df[cpu_filtered_df['hacking'] == 0]

print(f'''total size {len(unfiltered_df)} \n ''' +
      f'''hacks hand size {len(hacks_hand_filtered_df)}, legit hand size {len(legit_hand_filtered_df)} \n ''' +
      f'''hacks cpu size {len(hacks_cpu_filtered_df)}, legit cpu size {len(legit_cpu_filtered_df)}''')

# plot raw numbers
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

def plotNumWith100MSBins(df, col, ax):
    num_bins = math.ceil((df[col].max() - df[col].min()) / 0.2)
    df.hist(col, bins=num_bins, ax=ax)
    return pd.cut(df[col], num_bins).value_counts().sort_index()

hacks_hand_distribution = plotNumWith100MSBins(hacks_hand_filtered_df, 'hand_aim_react_s', ax[0][0])
hacks_cpu_distribution = plotNumWith100MSBins(hacks_cpu_filtered_df, 'cpu_aim_react_s', ax[1][0])
legit_hand_distribution = plotNumWith100MSBins(legit_hand_filtered_df, 'hand_aim_react_s', ax[0][1])
legit_cpu_distribution = plotNumWith100MSBins(legit_cpu_filtered_df, 'cpu_aim_react_s', ax[1][1])

for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax[i][j].set_xlim(-3, 3)
        if i == 0:
            ax[i][j].set_ylim(0, max(hacks_hand_distribution.max(), legit_hand_distribution.max()) + 2)
        else:
            ax[i][j].set_ylim(0, max(hacks_cpu_distribution.max(), legit_cpu_distribution.max()) + 1)
        ax[i][j].set_xlabel('Reaction Time (s)', fontsize=14)
        ax[i][j].set_ylabel('Frequency', fontsize=14)

ax[0][0].set_title('GPU Labeled, Hacking', fontsize=18)
ax[0][1].set_title('GPU Labeled, Not Hacking', fontsize=18)
ax[1][0].set_title('CPU Labeled, Hacking', fontsize=18)
ax[1][1].set_title('CPU Labeled, Not Hacking', fontsize=18)

ax[0][0].annotate('total points: ' + str(len(hacks_hand_filtered_df)), (1.1,45), fontsize="14")
ax[0][1].annotate('total points: ' + str(len(legit_hand_filtered_df)), (1.1,45), fontsize="14")
ax[1][0].annotate('total points: ' + str(len(hacks_cpu_filtered_df)), (1.1,11.2), fontsize="14")
ax[1][1].annotate('total points: ' + str(len(legit_cpu_filtered_df)), (1.1,11.2), fontsize="14")

plt.tight_layout()
fig.savefig(args.plot_folder + 'histogram__hand_vs_cpu__hacking_vs_legit.png')

plt.clf()

# plot percentages

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

def plotPctWith100MSBins(df, col, ax):
    num_bins = math.ceil((df[col].max() - df[col].min()) / 0.2)
    df.hist(col, bins=num_bins, ax=ax, weights=np.ones(len(df[col])) / len(df[col]))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    return pd.cut(df[col], num_bins).value_counts().sort_index()

plotPctWith100MSBins(hacks_hand_filtered_df, 'hand_aim_react_s', ax[0][0])
plotPctWith100MSBins(hacks_cpu_filtered_df, 'cpu_aim_react_s', ax[1][0])
plotPctWith100MSBins(legit_hand_filtered_df, 'hand_aim_react_s', ax[0][1])
plotPctWith100MSBins(legit_cpu_filtered_df, 'cpu_aim_react_s', ax[1][1])

# make rows share same ylim
max_row_0_ylim = max(ax[0][0].get_ylim()[1], ax[0][1].get_ylim()[1])
ax[0][0].set_ylim(0, max_row_0_ylim)
ax[0][1].set_ylim(0, max_row_0_ylim)

max_row_1_ylim = max(ax[1][0].get_ylim()[1], ax[1][1].get_ylim()[1])
ax[1][0].set_ylim(0, max_row_1_ylim)
ax[1][1].set_ylim(0, max_row_1_ylim)

for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax[i][j].set_xlim(-3, 3)
        ax[i][j].set_xlabel('Reaction Time (s)', fontsize=14)
        ax[i][j].set_ylabel('Percent Frequency', fontsize=14)

ax[0][0].set_title('GPU Labeled, Hacking', fontsize=18)
ax[0][1].set_title('GPU Labeled, Not Hacking', fontsize=18)
ax[1][0].set_title('CPU Labeled, Hacking', fontsize=18)
ax[1][1].set_title('CPU Labeled, Not Hacking', fontsize=18)
plt.suptitle('Individual Data Points Histograms', fontsize=30)
plt.tight_layout()
fig.savefig(args.plot_folder + 'percent_histogram__hand_vs_cpu__hacking_vs_legit.png')

plt.clf()

all_filtered_df = sqlio.read_sql_query("select * from react_final where hacking < 2 and abs(hand_aim_react_s) <= 3.0 and abs(cpu_aim_react_s) <= 3.0", conn)
#all_filtered_df = pd.concat([hand_filtered_df, cpu_filtered_df], ignore_index=True)
all_filtered_df['hacking'] = all_filtered_df['hacking'].map({True: 1, False: 0})
print(f'''all filtered size {len(all_filtered_df)}''')

for hand_or_cpu in ['hand', 'cpu']:
    plt.clf()
    X_df = all_filtered_df[[hand_or_cpu + '_aim_react_s', 'distinct_others_spotted_during_time']]
    y_series = all_filtered_df['hacking']

    X_train, X_test, y_train, y_test = train_test_split(X_df,y_series,test_size=0.2,random_state=42)

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.set(font_scale=1.5)
    confusion_matrix_heatmap = sn.heatmap(confusion_matrix, annot=True)
    plt.title(hand_or_cpu + ' Labeled Confusion Matrix', fontsize=24)
    confusion_matrix_figure = confusion_matrix_heatmap.get_figure()
    confusion_matrix_figure.savefig(args.plot_folder + hand_or_cpu + '_confusion_matrix__hand_vs_cpu__hacking_vs_legit.png')

    print(hand_or_cpu + ' coeff: ', lr_model.coef_)
    print(hand_or_cpu + ' accuracy: ', metrics.accuracy_score(y_test, y_pred))
    #print(np.argwhere(y_pred == False))
