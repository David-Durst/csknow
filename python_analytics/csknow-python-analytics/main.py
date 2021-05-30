from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


def clusterDataset(df, name):
    # cluster overall
    kmeans_columns = ['wall x', 'wall y', 'wall z']
    kmeans_df = df[kmeans_columns]
    kmeans_matrix = kmeans_df.values
    unpartitioned_kmeans = KMeans(n_clusters=24).fit(kmeans_matrix);
    #for cluster_center in kmeans.cluster_centers_:
    #    print("box " + str(cluster_center[0] - 20) + " " + str(cluster_center[1] - 20) + " "
    #          + str(cluster_center[2] - 20) + " "
    #          + str(cluster_center[0] + 20) + " " + str(cluster_center[1] + 20) + " "
    #          + str(cluster_center[2] + 20) + " ; ")


    clusters_f = open(name + "_clusters.csv", "w")
    csgo_f = open(name + "_clusters.cfg", "w")
    clusters_f.write("cluster id,wall id,cluster x, cluster y, cluster z\n")
    cluster_id = 0;
    for cluster_center in unpartitioned_kmeans.cluster_centers_:
        clusters_f.write(str(cluster_id) + ",-1," + str(cluster_center[0]) + "," + str(cluster_center[1]) + "," + str(cluster_center[2]) + "\n")
        cluster_id += 1
        csgo_f.write("box " + str(cluster_center[0] - 20) + " " + str(cluster_center[1] - 20) + " "
                 + str(cluster_center[2] - 20) + " "
                 + str(cluster_center[0] + 20) + " " + str(cluster_center[1] + 20) + " "
                 + str(cluster_center[2] + 20) + "\n")
    clusters_f.close()
    csgo_f.close()

# %%j
def heatmapByWall(df, wallDF, name):
    reasonableHeightDF = df[(df['wall z'] > -1000) & (df['wall z'] < 2000)]
    wallIDs = reasonableHeightDF['wall id'].unique()
    wallIDs.sort()
    for id in wallIDs:
        # why copy: https://www.dataquest.io/blog/settingwithcopywarning/'
        wallDF = df[df['wall id'] == id].copy()
        # pick if wall in x or y dimension
        xDelta = wallDF['wall x'].max() - wallDF['wall x'].min()
        xOrYGraphDim = 'wall x'
        if xDelta < 0.5:
            xOrYGraphDim = 'wall y'
        # grouping by x and z values in modded by 1
        print(wallDF)
        wallDF['wall z'] = wallDF['wall z'].apply(np.int64) // 100 * 100
        wallDF[xOrYGraphDim] = wallDF[xOrYGraphDim].apply(np.int64) // 100 * 100
        print(wallDF)
        heatmapDF = pd.pivot_table(wallDF, values='id', index=['wall z'], columns=[xOrYGraphDim], aggfunc='count', fill_value=0)
        print(heatmapDF)
        fig, ax = plt.subplots()
        sns.heatmap(heatmapDF, ax=ax)
        ax.invert_yaxis()
        ax.set_title(wallDF.iloc[id]['name'])
        plt.tight_layout()
        plt.savefig(name + '_' + wallDF.iloc[id]['name'] + '.png')
        plt.clf()


# %%


# load data and run scripts
aCatDF = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/a_cat_peekers.csv")
aCatWalls = pd.read_csv(os.getcwd() + "/../../analytics/walls/aCatWalls.csv")
clusterDataset(aCatDF, "a_cat_peekers")
heatmapByWall(aCatDF, "a_cat_peekers")
midTDF = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/mid_ct_peekers.csv")
midWalls = pd.read_csv(os.getcwd() + "/../../analytics/walls/midWalls.csv")
clusterDataset(midTDF, "mid_ct_peekers")
heatmapByWall(midTDF, "mid_ct_peekers")
