import numpy as np
from scipy.special import loggamma
from scipy.special import psi
import pandas as pd


class MultiViewCTR:


    def __init__(self, n_latent, alpha=1, beta=1):
        # number of latent variables
        self.n_latent = n_latent
        
        # Dirichlet "smoothing" hyperparameters 
        self.alpha = alpha
        self.beta = beta

        # nll to track convergence
        self.neg_log_likelihood = []


    def init_cluster_assignment(self, X):
        """
        Do random cluster assignment for datapoints.

        Parameters:
        X (ndarray): dataset.
        """
        # number of features
        self.n_features = X.shape[1]

        # p = None assumes a uniform distribution
        self.cluster_assignment = np.random.choice(
            self.n_latent, size=len(X), replace=True, p=None
        )

        # number of datapoints (or total ratings)
        # assigned to cluster-feature, shape (n_latent, n_features)
        _, counts = np.unique(self.cluster_assignment, return_counts=True)
        self.cluster_feature_count = np.tile(counts, (self.n_features, 1)).T

    def update(self, x, z, i):
        """
        Increment or decrement datapoint from the 
        latent variable to which it is assigned.

        Parameters:
        x (n_features,): one-hot encoded feature-vector.
        z (int): latent variable.
        i (int): amount by which to increment.

        Returns:
        
        Example:
        >>> update(z=1, x=[0, 0, 1], i=-1)
        """
        # update cluster assignment
        self.cluster_feature_count[z, :] += i*x

        # assert no negative counts
        assert not (self.cluster_feature_count < 0).any() 

    def get_log_prob(self):
        """
        Get log probabilities.

        Returns:
        Unnormalized log probability vector.
        """
        # cluster feature counts of shape (n_latent, n_features)
        m = self.cluster_feature_count
        # cluster assignment counts
        n = m.sum(axis=1)  

        # cluster log probability
        log_prob = np.log(n + self.alpha)
        # feature log probability
        q = m + self.beta
        log_prob += np.log(q).sum(axis=1)
        log_prob -= np.log(q.sum(axis=1))

        # assert no nans in log proba
        assert not np.isnan(log_prob).any()

        return log_prob
        
    def sample(self):
        """
        Sample form Markov chain.

        Returns:
        z_new (int): sampled latent variable.
        """
        # get unnormalized log probabilities
        log_prob = self.get_log_prob()
        
        # apply the log normalization trick
        prob = np.exp(log_prob - log_prob.max())
        prob /= sum(prob)

        # assert normalization
        assert round(sum(prob), 5) == 1, "Probability vector not normalized."

        # sample
        z_new = np.random.choice(self.n_latent, 1, p=prob)[0]
        return z_new

    def get_log_likelihood(self):
        """
        Compute log likelihood.
        """
        # feature log probability
        q = self.cluster_feature_count + self.beta
        log_likelihood = loggamma(q).sum(axis=1)
        log_likelihood -= loggamma(q.sum().sum())

        return log_likelihood

    def append_neg_log_likelihood(self):
        """
        Get neg log likelihood and append to trace.
        """
        # get log likelihood
        log_likelihood = self.get_log_likelihood()
        # append to trace
        self.neg_log_likelihood.append(-log_likelihood.sum())

    def fit(self, X, n_iter=5):
        """
        Sample from Markov chain.

        Parameters:
        n_iter (int): number of dataset iterations.
        """
        # creates self.cluster_assignment vector
        # of shape (n_datapoints,)
        self.init_cluster_assignment(X)

        self.trace = []
        for _ in range(n_iter):
            for t, x in enumerate(X):
                # append neg log likelihood
                self.append_neg_log_likelihood()
                # get current assignment
                z_old = self.cluster_assignment[t]
                # decrement 
                self.update(x, z_old, -1)  
                # sample latent variable
                z_new = self.sample()
                # increment
                self.update(x, z_new, +1)  
                # update assignment
                self.cluster_assignment[t] = z_new
                # append to trace
                self.trace.append(z_new)

    def transform(self, X, burn_in=1000):
        """
        Transform data into embeddings.

        Returns:
        Vector embeddings.
        """
        self.cluster_feature_count

    def predict(self, X, idx):
        """
        Predict embeddings for data.

        Parameters:
        X: (ndarray): dataset.
        idx (ndarray): index of datapoints.
        """
        # cluster counts
        _, cc = np.unique(self.trace, return_counts=True)
        cc = cc / sum(cc)

        #cluster feature counts
        cfc = self.cluster_feature_count
        cfc = cfc / cfc.sum(axis=0)

        # compute embeddings 
        embs = X @ (cc.reshape(-1, 1) * np.log(cfc)).T 
        embs += np.log(cc).mean(axis=0)

        # get embeddings for each datapoint
        data = pd.DataFrame(embs, index=idx)
        data = data.groupby(idx).mean()

        return np.exp(data).to_numpy()