"""clustering.py: Starter file for assignment on Clustering """

__author__ = "Shishir Shah"
__version__ = "1.0.0"
__copyright__ = "All rights reserved.  This software  \
                should not be distributed, reproduced, or shared online, without the permission of the author."

# Data Manipulation and Visualization
import pandas as pd #creating and manipulating dataframes
import matplotlib.pyplot as plt #visuals
import seaborn as sns #visuals
from sklearn.cluster import KMeans #K-Means
from sklearn.cluster import DBSCAN #DBSCAN
from sklearn.metrics import confusion_matrix
import numpy as np


__author__ = "Josh Aneke"
__version__ = "1.1.0"

'''
Github Username: JoshAneke
PSID: 1828214
'''

# Reading the data
data = pd.read_csv('data/clinical_records_dataset.csv')
class_labels = data['DEATH_EVENT']
data = data.drop('DEATH_EVENT', axis=1)
data = data.drop('time', axis=1)
cluster_labels = data[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                       'ejection_fraction', 'high_blood_pressure', 'platelets',
                       'serum_creatinine', 'serum_sodium', 'sex', 'smoking']]

''' Write your code here '''
'''#1'''
def purity(class_labels, cluster_labels):
    # Create a dictionary to store the correspondence between true labels and cluster labels
    label_mapping = {}

    # Zip true_labels and cluster_labels together and iterate over them
    for true_label, cluster_label in zip(class_labels, cluster_labels):
        # Create a tuple (class_label, cluster_label)
        pair = (class_labels, cluster_label)

        # If the pair is already in the dictionary, increment its count
        if pair in label_mapping:
            label_mapping[pair] += 1
        else:
            # Otherwise, initialize the count to 1
            label_mapping[pair] = 1

    # Initialize the purity score
    total_correct = 0

    # Iterate over unique cluster labels
    for cluster_label in set(cluster_labels):
        # Find the true label with the maximum count for the current cluster label
        max_true_label = max((class_labels for class_labels, current_cluster_label
                              in label_mapping if current_cluster_label == cluster_label),
                             key=lambda x: label_mapping[(x, cluster_label)])

        # Add the count of the most common true label to the total_correct
        total_correct += label_mapping[(max_true_label, cluster_label)]

    # Compute purity as the total_correct divided by the total number of data points
    purity_score = total_correct / len(class_labels)

    return purity_score


'''#2'''
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_label = kmeans.fit_predict(cluster_labels)

# Combining the cluster labels with class labels
result_df = pd.DataFrame({'Death Event': class_labels, 'Cluster': cluster_label})

# Creating a confusion matrix
conf_matrix = confusion_matrix(result_df['Death Event'], result_df['Cluster'])

# Computing purity for each cluster
purity_per_cluster = np.max(conf_matrix, axis=0) / np.sum(conf_matrix, axis=0)

# Computing overall purity
overall_purity = np.sum(np.max(conf_matrix, axis=0)) / np.sum(conf_matrix)

# Displaying results
print("Purity for Cluster 0:", purity_per_cluster[0])
print("Purity for Cluster 1:", purity_per_cluster[1])
print("Overall Purity:", overall_purity)

# Finding the cluster with the highest purity
highest_purity_cluster = np.argmax(purity_per_cluster)
print("Cluster with the Highest Purity:", highest_purity_cluster)

# Percentage of data points assigned to the highest purity cluster
percentage_highest_purity_cluster = np.sum(conf_matrix[:, highest_purity_cluster]) / np.sum(conf_matrix)
print("Percentage of Data Points in the Highest Purity Cluster:", percentage_highest_purity_cluster)

# Percentage of data points assigned to the other cluster
percentage_other_cluster = 1 - percentage_highest_purity_cluster
print("Percentage of Data Points in the Other Cluster:", percentage_other_cluster)

'''#3'''
num_clusters2 = 3
kmeans2 = KMeans(n_clusters=num_clusters2, random_state=42)
cluster_labels2 = kmeans2.fit_predict(cluster_labels)

# Combining the cluster labels with class labels
result_df2 = pd.DataFrame({'Death Event': class_labels, 'Cluster2': cluster_labels2})

# Creating a confusion matrix
conf_matrix2 = confusion_matrix(result_df2['Death Event'], result_df2['Cluster2'])

# Computing purity for each cluster
purity_per_cluster2 = np.max(conf_matrix2, axis=0) / np.sum(conf_matrix2, axis=0)

# Computing overall purity
overall_purity2 = np.sum(np.max(conf_matrix2, axis=0)) / np.sum(conf_matrix2)

# Displaying results
print("Purity for Cluster 0:", purity_per_cluster2[0])
print("Purity for Cluster 1:", purity_per_cluster2[1])
print("Purity for Cluster 2:", purity_per_cluster2[2])
print("Overall Purity:", overall_purity2)

# Finding the cluster with the highest purity
highest_purity_cluster2 = np.argmax(purity_per_cluster2)
print("Cluster with the Highest Purity:", highest_purity_cluster2)

# Percentage of data points assigned to the highest purity cluster
percentage_highest_purity_cluster2 = np.sum(conf_matrix2[:, highest_purity_cluster2]) / np.sum(conf_matrix2)
print("Percentage of Data Points in the Highest Purity Cluster:", percentage_highest_purity_cluster2)


num_clusters3 = 5
kmeans3 = KMeans(n_clusters=num_clusters3, random_state=42)
cluster_labels3 = kmeans3.fit_predict(cluster_labels)

# Combining the cluster labels with class labels
result_df3 = pd.DataFrame({'Death Event': class_labels, 'Cluster3': cluster_labels3})

# Creating a confusion matrix
conf_matrix3 = confusion_matrix(result_df3['Death Event'], result_df3['Cluster3'])

# Computing purity for each cluster
purity_per_cluster3 = np.max(conf_matrix3, axis=0) / np.sum(conf_matrix3, axis=0)

# Computing overall purity
overall_purity3 = np.sum(np.max(conf_matrix3, axis=0)) / np.sum(conf_matrix3)

# Displaying results
print("Purity for Cluster 0:", purity_per_cluster3[0])
print("Purity for Cluster 1:", purity_per_cluster3[1])
print("Purity for Cluster 2:", purity_per_cluster3[2])
print("Purity for Cluster 3:", purity_per_cluster3[3])
print("Purity for Cluster 4:", purity_per_cluster3[4])
print("Overall Purity:", overall_purity3)

# Finding the cluster with the highest purity
highest_purity_cluster3 = np.argmax(purity_per_cluster3)
print("Cluster with the Highest Purity:", highest_purity_cluster3)

# Percentage of data points assigned to the highest purity cluster
percentage_highest_purity_cluster3 = np.sum(conf_matrix3[:, highest_purity_cluster3]) / np.sum(conf_matrix3)
print("Percentage of Data Points in the Highest Purity Cluster:", percentage_highest_purity_cluster3)

print("\nConsidering the following k-means, I think that k-means = 2 would be the best kmeans out. One reason would be that kmeans gets worse and worse to support higher data the higher the k-means is. Another reason is the the overall purity for the two clusters. Since the average for the two clusters combined is higher than the 3 and 5 clusters, it would be the most reasonable choice")

'''#4'''
min_pts = 5
eps = 0.5
dbscan = DBSCAN(eps=eps, min_samples=min_pts)
dbscan_labels = dbscan.fit_predict(cluster_labels)

# Combining the DBSCAN labels with class labels
result_df_dbscan = pd.DataFrame({'Death Event': class_labels, 'DBSCAN Cluster': dbscan_labels})

# Creating a confusion matrix for DBSCAN
conf_matrix_dbscan = confusion_matrix(result_df_dbscan['Death Event'], result_df_dbscan['DBSCAN Cluster'])

# Computing purity for each DBSCAN cluster
purity_per_cluster_dbscan = np.max(conf_matrix_dbscan, axis=0) / np.sum(conf_matrix_dbscan, axis=0)

# Computing overall purity for DBSCAN
overall_purity_dbscan = np.sum(np.max(conf_matrix_dbscan, axis=0)) / np.sum(conf_matrix_dbscan)

# Displaying results for DBSCAN
print("Purity for DBSCAN Cluster 0:", purity_per_cluster_dbscan[0])
print("Purity for DBSCAN Cluster 1:", purity_per_cluster_dbscan[1])
print("Overall Purity for DBSCAN:", overall_purity_dbscan)

# Finding the DBSCAN cluster with the highest purity
highest_purity_cluster_dbscan = np.argmax(purity_per_cluster_dbscan)
print("DBSCAN Cluster with the Highest Purity:", highest_purity_cluster_dbscan)

# Percentage of data points assigned to the highest purity DBSCAN cluster
percentage_highest_purity_cluster_dbscan = np.sum(conf_matrix_dbscan[:, highest_purity_cluster_dbscan]) / np.sum(conf_matrix_dbscan)
print("Percentage of Data Points in the Highest Purity DBSCAN Cluster:", percentage_highest_purity_cluster_dbscan)

# Percentage of data points assigned to the other DBSCAN cluster
percentage_other_cluster_dbscan = 1 - percentage_highest_purity_cluster_dbscan
print("Percentage of Data Points in the Other DBSCAN Cluster:", percentage_other_cluster_dbscan)

'''#5'''
minPts_range = range(2, 100)
eps_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

best_purity = 0.0
best_minPts = 0
best_eps = 0.0

for minPts2 in minPts_range:
    for eps2 in eps_range:
        # Run DBSCAN
        dbscan2 = DBSCAN(eps=eps2, min_samples=minPts2)
        dbscan_labels2 = dbscan2.fit_predict(cluster_labels)

        # Evaluate constraints
        unique_clusters = len(set(dbscan_labels2)) - 1  # Subtract 1 to exclude outliers
        percentage_outliers = np.sum(dbscan_labels2 == -1) / len(data)

        # Check constraints
        if 2 <= unique_clusters <= 18 and percentage_outliers < 0.1:
            conf_matrix_dbscan2 = confusion_matrix(class_labels, dbscan_labels2)
            
            if np.sum(conf_matrix_dbscan2) > 0:
                # Use a different variable name for DBSCAN purity
                purity_dbscan = np.sum(np.max(conf_matrix_dbscan2, axis=0)) / np.sum(conf_matrix_dbscan2)

                # Update best parameters if DBSCAN purity is higher
                if purity_dbscan > best_purity:
                    best_purity = purity_dbscan  # Use the correct variable here
                    best_minPts = minPts2
                    best_eps = eps2

# Output best parameters and purity
print("Best minPts:", best_minPts)
print("Best eps:", best_eps)
print("Best Purity:", best_purity)