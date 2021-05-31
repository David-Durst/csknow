import numpy as np
import pandas as pd
import os
from itertools import combinations


def sameSequence(df, i, j):
    return df.iloc[i]['round id'] == df.iloc[j]['round id'] and \
        df.iloc[i]['player id'] == df.iloc[j]['player id']

def generateSequenceBags(df, ngramLength=2):
    sequenceBags = []
    for i in range(len(df['id'])):
        sequenceBags.append([])
        for l in range(ngramLength):
            index = i - l
            if index < 0 or not sameSequence(df, i, index):
                sequenceBags[len(sequenceBags) - 1].append(-1)
            else:
                sequenceBags[len(sequenceBags) - 1].append(df.iloc[index]['cluster id'])
        sequenceBags[len(sequenceBags) - 1] = sorted(sequenceBags[len(sequenceBags) - 1])
    df['sequence bags'] = sequenceBags

def generateTransitiionMatrices(sequencesDF, clustersDF):
    combinations(clustersDF['cluster id'])


aCatSequences = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/a_cat_cluster_sequence.csv")
aCatClusters = pd.read_csv(os.getcwd() + "/a_cat_peekers_clusters.csv")
generateSequenceBags(aCatSequences, "a_cat_cluster_sequence")
