import numpy as np
import pandas as pd
import os
from itertools import combinations_with_replacement


def sameSequence(df, i, j):
    return df.iloc[i]['round id'] == df.iloc[j]['round id'] and \
        df.iloc[i]['player id'] == df.iloc[j]['player id']

# %%
def generateSequenceBags(df, ngramLength=2):
    #sequenceBags = []
    #for i in range(len(df['id'])):
    #    sequenceBag = []
    #    for l in range(ngramLength):
    #        index = i - l
    #        if index < 0 or not sameSequence(df, i, index):
    #            sequenceBag.append(-1)
    #        else:
    #            sequenceBag.append(df.iloc[index]['cluster id'])
    #    sequenceBags.append(sorted(sequenceBag))
    columnsToMerge = ['cluster id']
    for i in range(1,ngramLength):
        df['cluster id' + str(i)] = df['cluster id'].copy().shift(periods=i, fill_value=-1).apply(np.int64)
        columnsToMerge.append('cluster id' + str(i))
    df['sequence bags'] = df[columnsToMerge].apply(lambda row: tuple(sorted(row.values)), axis=1)

# %%
def generateTransitionMatrix(sequencesDF, clustersDF, ngramLength=2):
    cluster_ids = clustersDF['cluster id'].tolist()
    cluster_ids.insert(0, -1)
    cluster_id_combinations = [t for t in combinations_with_replacement(cluster_ids, ngramLength)]
    transition_matrix = pd.DataFrame(np.zeros([len(cluster_id_combinations), len(cluster_id_combinations)]),
                                     columns=cluster_id_combinations, index=cluster_id_combinations)

    sequencesDF['prior sequence bags'] = sequencesDF['sequence bags'].copy().shift(periods=1)
    sequencesDF.at[0,'prior sequence bags'] = cluster_id_combinations[0]
    transition_df = sequencesDF.groupby(['sequence bags', 'prior sequence bags']).agg({'cluster id': 'count'})
    #transition_index_list = transition_df.index.tolist()
    transition_matrix = pd.pivot_table(sequencesDF, values='cluster id', index=['sequence bags'], columns=['prior sequence bags'], aggfunc='count',
                               fill_value=0)
    #for (i,cur_bag) in enumerate(cluster_id_combinations):
    #    print("handling: " + str(cur_bag))
    #    for (j,old_bag) in enumerate(cluster_id_combinations):
    #        if (cur_bag, old_bag) in transition_index_list:
    #            # lookup by index fails, so get entry by index in indices
    #            k = transition_index_list.index((cur_bag, old_bag))
    #            transition_matrix.iloc[i,j] = transition_df.iloc[k]['cluster id']
    return (transition_df, transition_matrix)
    #print("hi")
    #for i in range(len(sequencesDF['sequence bags'])):
    #    if i > 0 and sameSequence(sequencesDF, i, i-1):
    #        # columns (first index) are to, rows are from
    #        transition_matrix[sequencesDF.iloc[i]['sequence bags']][sequencesDF.iloc[i-1]['sequence bags']] += 1
    #    else:
    #        transition_matrix[sequencesDF.iloc[i]['sequence bags']][cluster_id_combinations[0]] += 1

    #print(transition_matrix)

# %%
aCatSequences = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/a_cat_cluster_sequence.csv")
aCatClusters = pd.read_csv(os.getcwd() + "/a_cat_peekers_clusters.csv")

# %%
generateSequenceBags(aCatSequences)
transition_df, transition_matrix = generateTransitionMatrix(aCatSequences, aCatClusters)

