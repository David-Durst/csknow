from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import os
from os import path
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


def clusterDataset(df, name):
    # cluster overall
    kmeans_columns = ['wall x', 'wall y', 'wall z']
    kmeans_df = df[kmeans_columns]
    kmeans_matrix = kmeans_df.values
    unpartitioned_kmeans = MiniBatchKMeans(n_clusters=24).fit(kmeans_matrix)
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

# %%
def heatmapByWall(df, wallDF, name, screenshotFolder):
    reasonableHeightDF = df[(df['wall z'] > -1000) & (df['wall z'] < 2000)]
    wallIDs = reasonableHeightDF['wall id'].unique()
    wallIDs.sort()
    for id in wallIDs:
        # why copy: https://www.dataquest.io/blog/settingwithcopywarning/'
        perWallDF = df[df['wall id'] == id].copy()
        # pick if wall in x or y dimension
        xDelta = perWallDF['wall x'].max() - perWallDF['wall x'].min()
        xOrYGraphDim = 'wall x'
        if xDelta < 0.5:
            xOrYGraphDim = 'wall y'
        # grouping by x and z values in modded by 1
        perWallDF['wall z'] = perWallDF['wall z'].apply(np.int64) // 10 * 10
        perWallDF[xOrYGraphDim] = perWallDF[xOrYGraphDim].apply(np.int64) // 10 * 10
        heatmapDF = pd.pivot_table(perWallDF, values='id', index=['wall z'], columns=[xOrYGraphDim], aggfunc='count', fill_value=0)
        fig, ax = plt.subplots()
        sns.heatmap(heatmapDF, ax=ax)
        ax.invert_yaxis()
        ax.set_title(wallDF.iloc[id]['name'])
        if wallDF.iloc[id]['flip'] == 1:
            ax.invert_xaxis()
        plt.tight_layout()
        imgPath = name + '_' + wallDF.iloc[id]['name']
        plt.savefig(imgPath + ".png")
        plt.close(fig)
        if path.exists(screenshotFolder + "/" + wallDF.iloc[id]['name'] + ".png"):
            topIm = Image.open(imgPath + ".png")
            botIm = Image.open(screenshotFolder + "/" + wallDF.iloc[id]['name'] + ".png")
            dstIm = Image.new('RGB', (topIm.width, 2*topIm.height))
            botIm.thumbnail([topIm.width, topIm.height], Image.ANTIALIAS)
            dstIm.paste(topIm, (0,0))
            dstIm.paste(botIm, (0,topIm.height))
            dstIm.save(imgPath + "_complete.png")
        else:
            shutil.copy(imgPath + ".png", imgPath + "_complete.png")


# %%


# load data and run scripts
aCatDF = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/a_cat_peekers.csv")
aCatWalls = pd.read_csv(os.getcwd() + "/../../analytics/walls/aCatWalls.csv")
clusterDataset(aCatDF, "a_cat_peekers")
heatmapByWall(aCatDF, aCatWalls, "a_cat_peekers", os.getcwd() + "/../../analytics/walls/wallImages/aCat")
midTDF = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/mid_ct_peekers.csv")
midWalls = pd.read_csv(os.getcwd() + "/../../analytics/walls/midWalls.csv")
clusterDataset(midTDF, "mid_ct_peekers")
heatmapByWall(midTDF, midWalls, "mid_ct_peekers", os.getcwd() + "/../../analytics/walls/wallImages/midCT")
