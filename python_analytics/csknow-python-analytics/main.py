from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os


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


# load data and run scripts
aCatDF = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/a_cat_peekers.csv")
clusterDataset(aCatDF, "a_cat_peekers")
midTDF = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/mid_t_peekers.csv")
clusterDataset(midTDF, "mid_t_peekers")
