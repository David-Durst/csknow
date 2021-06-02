import numpy as np
import pandas as pd
import csv
import os
from itertools import combinations_with_replacement


def sameSequence(df, i, j):
    return df.iloc[i]['round id'] == df.iloc[j]['round id'] and \
        df.iloc[i]['player id'] == df.iloc[j]['player id']

# %%
def generateSequenceBags(df, ngramLength=2):
    columnsToMerge = ['cluster id']
    for i in range(1,ngramLength):
        df['cluster id' + str(i)] = df['cluster id'].copy().shift(periods=i, fill_value=-1).apply(np.int64)
        columnsToMerge.append('cluster id' + str(i))
    df['sequence bags'] = df[columnsToMerge].apply(lambda row: tuple(sorted(row.values)), axis=1)

# %%
def generateTransitionMatrix(sequencesDF, dstFolder):
    sequencesDF['prior sequence bags'] = sequencesDF['sequence bags'].copy().shift(periods=1)
    sequencesDF.at[0,'prior sequence bags'] = (-1, -1)
    transition_df = sequencesDF.groupby(['sequence bags', 'prior sequence bags']).agg({'cluster id': 'count'})
    transition_matrix = pd.pivot_table(sequencesDF, values='cluster id', index=['sequence bags'], columns=['prior sequence bags'], aggfunc='count',
                               fill_value=0)
    filtered_matrix = transition_matrix.loc[(transition_matrix != 0).any(axis=1)]
    sorted_df = transition_df.sort_values('cluster id', 0, ascending=False)
    sorted_df.to_csv(dstFolder + "/transition_df.csv", sep=";")
    filtered_matrix.to_csv(dstFolder + "/transition_matrix.csv", sep=";")

# %%
aCatSequences = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/a_cat_cluster_sequence.csv")
generateSequenceBags(aCatSequences,1)
generateTransitionMatrix(aCatSequences, os.getcwd() + "/../transitionMatrices/aCat/")

midCTSequences = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/mid_ct_cluster_sequence.csv")
generateSequenceBags(midCTSequences,1)
generateTransitionMatrix(midCTSequences, os.getcwd() + "/../transitionMatrices/midCT/")
