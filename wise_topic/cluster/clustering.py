import hdbscan
import numpy as np
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


# Do the clustering in higher dims, plot in lower dims
def WithProbabilities(a: type):
    class WithProbabilities(a):
        def fit(self, X):
            super().fit()
            try:
                self.probabilities_ = self.predict_proba(X)
            except:
                labels = np.unique(self.labels_)
                self.probabilities_ = np.zeros((len(X), len(labels)))
                inds = np.array(enumerate(labels))
                self.probabilities_[inds] = 1.0


def agglomerative_clustering(X, n_clusters=10, **kwargs):
    clustering = AgglomerativeClustering(linkage="ward", n_clusters=n_clusters, **kwargs)
    clustering.fit(X)
    return clustering


def kmeans_clustering(dataset, max_clusters=10, n_clusters=None, **kwargs):
    best_score = -1
    best_clusters = None

    if n_clusters is None:
        for n_clusters in range(3, max_clusters + 1):
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, **kwargs)
            labels = kmeans.fit_predict(dataset)

            # Calculate silhouette score
            score = silhouette_score(dataset, labels)
            print(n_clusters, score)

            # Update best score and clusters if current score is better
            if score > best_score:
                best_score = score
                best_clusters = kmeans
        return best_clusters
    else:
        kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        kmeans.fit(dataset)
        return kmeans


def bayesian_gaussian_mixture(X, prior: float = 0.01, **kwargs):
    bgmm = mixture.BayesianGaussianMixture(
        n_components=10,
        covariance_type="full",
        weight_concentration_prior=prior,
        weight_concentration_prior_type="dirichlet_process",
        mean_precision_prior=1e-2,
        # covariance_prior=1e0 * np.eye(x.shape),
        init_params="kmeans",
        max_iter=100,
        random_state=2,
        **kwargs
    ).fit(X)
    bgmm.labels_ = bgmm.predict(X)
    return bgmm


def hdbscan_clustering(X, min_topic_size=10, **kwargs):
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        **kwargs
    )
    hdb.fit(X)
    return hdb
