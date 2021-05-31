import numpy as np
import pandas as pd
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
def generateTransitionMatrix(sequencesDF):
    sequencesDF['prior sequence bags'] = sequencesDF['sequence bags'].copy().shift(periods=1)
    sequencesDF.at[0,'prior sequence bags'] = (-1, -1)
    transition_df = sequencesDF.groupby(['sequence bags', 'prior sequence bags']).agg({'cluster id': 'count'})
    transition_matrix = pd.pivot_table(sequencesDF, values='cluster id', index=['sequence bags'], columns=['prior sequence bags'], aggfunc='count',
                               fill_value=0)
    return (transition_df, transition_matrix)

    #print(transition_matrix)

# %%
aCatSequences = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/a_cat_cluster_sequence.csv")
aCatClusters = pd.read_csv(os.getcwd() + "/a_cat_peekers_clusters.csv")

# %%
generateSequenceBags(aCatSequences)
transition_df, transition_matrix = generateTransitionMatrix(aCatSequences)

