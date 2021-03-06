import SkeletonSplitAndMerge as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import arviz as az

if __name__ == '__main__':
	data = np.genfromtxt("galaxy_dataset.txt", delimiter=',')

	sns.scatterplot(x=data, y=np.repeat(1, len(data)), linewidth=0)
	#sns.histplot(x=data)
	#sns.kdeplot(x=data)
	plt.yticks(ticks=[])
	plt.savefig("data.png", dpi=500)
	plt.close()

	burn_in = 1000
	n_iterations = 5000
	thinning_step = 2
	labels = np.full(len(data), 1, dtype=int)
	#prior_distribution = sm.NNIGHierarchy(25, 0.08, 4, 8, 8)
	prior_distribution = sm.NNIGHierarchy(25, 0.08, 4, 8, 0.75)

	labels_samples = sm.SplitAndMerge(data, labels, prior_distribution).\
        SplitAndMergeAlgo(5, 1, 1, N=n_iterations)

	labels_samples = np.delete(labels_samples, range(burn_in), 0)

	labels_samples = np.delete(labels_samples, range(0,labels_samples.shape[0],\
		thinning_step), 0)

	n_clusters_samples = np.apply_along_axis(lambda x: len(np.unique(x)), 1,\
  	labels_samples)
	p = sns.lineplot(x=np.full(labels_samples.shape[0], 1, dtype=int).cumsum(),\
  	y=n_clusters_samples)
	plt.xlabel("Iteration")
	plt.ylabel("Number of clusters")
	plt.savefig("n_clusters.png", dpi=500)
	plt.close()

	clust_estimate = sm.cluster_estimate(labels_samples)
	clust_estimate = ss.rankdata(clust_estimate, method='dense')

	sns.scatterplot(x=data, y=np.repeat("Split&Merge", len(data)),\
		hue=clust_estimate,palette = "tab10", linewidth=0, legend=None)
	#sns.kdeplot(data, hue=clust_estimate, palette="tab10")
	plt.yticks(rotation=90, verticalalignment="center")
	plt.savefig("data_clustered.png", dpi=500)
	plt.close()

	sns.kdeplot(data, hue=clust_estimate, palette="tab10")
	plt.savefig("data_clustered_kde.png", dpi=500)
	plt.close()

	print(az.ess(n_clusters_samples))

	np.savetxt("cluster_labels.txt", clust_estimate, delimiter=',')