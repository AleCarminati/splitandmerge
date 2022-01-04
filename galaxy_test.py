import SkeletonSplitAndMerge as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

if __name__ == '__main__':
	data = np.genfromtxt("galaxy_dataset.txt", delimiter=',')

	sns.scatterplot(x=data, y=np.repeat(1, len(data)))
	#sns.histplot(x=data)
	#sns.kdeplot(x=data)
	plt.savefig("data.png")
	plt.close()

	n_iterations = 1500
	labels = np.full(len(data), 1, dtype=int)
	prior_distribution = sm.NNIGHierarchy(25, 0.08, 4, 8, 8)

	labels_samples = sm.SplitAndMerge(data, labels, prior_distribution).\
        SplitAndMergeAlgo(5, 1, 1, N=n_iterations)

	n_clusters_samples = np.apply_along_axis(lambda x: len(np.unique(x)), 1,\
  	labels_samples)
	p = sns.lineplot(x=np.full(n_iterations, 1, dtype=int).cumsum(),\
  	y=n_clusters_samples)
	plt.xlabel("Iteration")
	plt.ylabel("Number of clusters")
	plt.savefig("n_clusters.png")
	plt.close()

	clust_estimate = sm.cluster_estimate(labels_samples)
	clust_estimate = ss.rankdata(clust_estimate, method='dense')

	sns.scatterplot(x=data, y=np.repeat(0.01, len(data)), hue=clust_estimate,\
		palette = "tab10")
	sns.kdeplot(data, hue=clust_estimate, palette="tab10")
	plt.savefig("data_clustered.png")
	plt.close()