# %%
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os

# %%
# load data
df = pd.read_csv(os.getcwd() + "/../../analytics/csv_outputs/23_05_2021__14_43_55_a_cat_peekers.csv")

# %%
# partition it by wall
walls = np.sort(df['wall id'].unique())
partitioned_dfs = []
for wall_id in walls:
    partitioned_dfs.append(df.loc[df['wall id'] == wall_id])

# %%
# cluster per partition
kmeans_per_wall = []
for partitioned_df in partitioned_dfs:
    kmeans_columns = ['wall x', 'wall y', 'wall z']
    kmeans_df = partitioned_df[kmeans_columns]
    kmeans_matrix = kmeans_df.values
    kmeans = KMeans(n_clusters=3).fit(kmeans_matrix);
    kmeans_per_wall.append(kmeans)
#    print("for wall " + str(partitioned_df.iloc[0,6]))
#    for cluster_center in kmeans.cluster_centers_:
#        print("box " + str(cluster_center[0] - 20) + " " + str(cluster_center[1] - 20) + " "
#              + str(cluster_center[2] - 20) + " "
#              + str(cluster_center[0] + 20) + " " + str(cluster_center[1] + 20) + " "
#              + str(cluster_center[2] + 20) + " ; ")

# %%
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

# %%
partitioned_clusters_f = open("partitioned_a_cat_peekers_clusters.csv", "w")
partitioned_csgo_f = open("partitioned_a_cat_peekers.cfg", "w")
partitioned_clusters_f.write("cluster id,wall id,cluster x, cluster y, cluster z\n")
cluster_id = 0;
for i in range(len(kmeans_per_wall)):
    kmeans = kmeans_per_wall[i]
    wall_id = walls[i]
    for cluster_center in kmeans.cluster_centers_:
        partitioned_clusters_f.write(str(cluster_id) + "," + str(wall_id) + "," + str(cluster_center[0]) + "," + str(cluster_center[1]) + "," + str(cluster_center[2]) + "\n")
        cluster_id += 1
        partitioned_csgo_f.write("box " + str(cluster_center[0] - 20) + " " + str(cluster_center[1] - 20) + " "
              + str(cluster_center[2] - 20) + " "
              + str(cluster_center[0] + 20) + " " + str(cluster_center[1] + 20) + " "
              + str(cluster_center[2] + 20) + "\n")
partitioned_clusters_f.close()
partitioned_csgo_f.close()

# %%
clusters_f = open("a_cat_peekers_clusters.csv", "w")
csgo_f = open("a_cat_peekers.cfg", "w")
clusters_f.write("cluster id,cluster x, cluster y, cluster z\n")
cluster_id = 0;
for cluster_center in unpartitioned_kmeans.cluster_centers_:
    clusters_f.write(str(cluster_id) + "," + str(cluster_center[0]) + "," + str(cluster_center[1]) + "," + str(cluster_center[2]) + "\n")
    cluster_id += 1
    csgo_f.write("box " + str(cluster_center[0] - 20) + " " + str(cluster_center[1] - 20) + " "
             + str(cluster_center[2] - 20) + " "
             + str(cluster_center[0] + 20) + " " + str(cluster_center[1] + 20) + " "
             + str(cluster_center[2] + 20) + "\n")
clusters_f.close()
csgo_f.close()
