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
from plottingHelpers import *
from regressionHelpers import *
from dataframeHelpers import *

parser = argparse.ArgumentParser()
parser.add_argument("password", help="database password",
                    type=str)
parser.add_argument("query_file", help="file containing query",
                    type=str)
parser.add_argument("plot_folder", help="folder_for_plots",
                    type=str)
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
base_select_str = "select * from react_final where abs(aim_react_s) <= 3 and not seen_last_five_seconds"
filtered_df = sqlio.read_sql_query(base_select_str, conn)

filtered_df[filtered_df['fire_react_s'] > 2] = np.NaN

dfs = VisibilityTechniqueDataFrames(unfiltered_df, filtered_df)
dfs.print_size()


makeHistograms(dfs.get_as_grid(), 'aim_react_s', makePlotterFunction(0.2, False), plot_titles,
               'Individual Count Aim Reactions', 'Aim Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'aim_react_s', makePlotterFunction(0.2, True), plot_titles,
               'Individual Percent Aim Reactions', 'Aim Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'fire_react_s', makePlotterFunction(0.5, False), plot_titles,
               'Individual Count Fire Reactions', 'Fire Reaction Time (s)', args.plot_folder)
makeHistograms(dfs.get_as_grid(), 'fire_react_s', makePlotterFunction(0.5, True), plot_titles,
               'Individual Percent Fire Reactions', 'Fire Reaction Time (s)', args.plot_folder)


individual_str = 'individual'
makeLogReg(dfs.pix_adjusted_dfs.get_hacks_union_legit(), ['aim_react_s'], visibility_techniques[0], individual_str, args.plot_folder)
makeLogReg(dfs.pix_unadjusted_dfs.get_hacks_union_legit(), ['aim_react_s'], visibility_techniques[1], individual_str, args.plot_folder)
makeLogReg(dfs.bbox_dfs.get_hacks_union_legit(), ['aim_react_s'], visibility_techniques[2], individual_str, args.plot_folder)
#makeLogReg(cpu_filtered_df, ['avg_aim_cpu_react'], 'CPU')
#makeLogReg(cpu_just_legit_cheat_df, ['avg_aim_cpu_react', 'avg_fire_cpu_react', 'cpu_preaims'], 'BBox')
#all_filtered_df = pd.concat([hand_filtered_df, cpu_filtered_df], ignore_index=True)

#for hand_or_cpu in ['hand', 'cpu']:
#    plt.clf()
#    X_df = all_filtered_df[[hand_or_cpu + '_aim_react_s', 'distinct_others_spotted_during_time']]
#    y_series = all_filtered_df['hacking']
#
#    X_train, X_test, y_train, y_test = train_test_split(X_df,y_series,test_size=0.2,random_state=42)
#
#    lr_model = LogisticRegression()
#    lr_model.fit(X_train, y_train)
#    y_pred = lr_model.predict(X_test)
#
#    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
#    sn.set(font_scale=1.5)
#    confusion_matrix_heatmap = sn.heatmap(confusion_matrix, annot=True)
#    plt.title(hand_or_cpu + ' Labeled Confusion Matrix', fontsize=24)
#    confusion_matrix_figure = confusion_matrix_heatmap.get_figure()
#    confusion_matrix_figure.savefig(args.plot_folder + hand_or_cpu + '_confusion_matrix__hand_vs_cpu__hacking_vs_legit.png')
#
#    print(hand_or_cpu + ' coeff: ', lr_model.coef_)
#    print(hand_or_cpu + ' accuracy: ', metrics.accuracy_score(y_test, y_pred))
#    #print(np.argwhere(y_pred == False))
