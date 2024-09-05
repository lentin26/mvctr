import numpy as np
import pandas as pd


class replayer:

    def __init__(self, model):
        # model used to create embeddings
        self.model = model

        # init objects
        self.user_support = {}
        self.item_support = {}
        self.user_embs = {}
        self.item_embs = {}
    
    def fit(self, X, y):
        """
        Get user and item embeddings.
        """
        # transform data
        X = X.multiply(y, axis=0)
        # fit embedding model
        self.model.fit(X.to_numpy())
        # user embeddings
        self.user_embs, self.user_cs_emb = self._get_embeddings(X, 'user_id')
        # item embeddings
        self.item_embs, self.item_cs_emb = self._get_embeddings(X, 'item_id')
        # update user support
        self.user_support = self.get_user_support(X)
        # update item support
        self.item_support = self.get_item_support(X)

    def _get_embeddings(self, X, index_name):
        """
        Get user and item embeddings and cold-start embeddings.

        param; X: array of (n_ratings, n_features)
            embedding for each rating event.
        """
        # get transformed dataset
        embs = pd.DataFrame(self.model.transform(X.to_numpy()))
        # add index back in
        embs.index = X.index

        # get all embeddings as a list of lists for each user
        embs = embs.groupby(
            embs.index.get_level_values(index_name)
        ).apply(lambda x: x.values.tolist()) 

        # avg embs for users and avg all to get cold-start embedding
        cs_emb = embs.apply(lambda x: np.mean(x, axis=0)).mean().tolist()

        # include cold-start embeddings and avg user embeddings
        embs = embs.apply(lambda x: np.mean(x + [cs_emb], axis=0).tolist()).to_dict()

        return embs, cs_emb

    def init_user_embeddings(self, ids):
        """
        Update embeddings for users using cold-start when needed.
        """
        self.user_embs = {
            id: self.user_embs[id] if id in self.user_embs
            else self.user_cs_emb for id in ids
        }

    def init_item_embeddings(self, ids):
        """
        Update embeddings for items using cold-start when needed.
        """
        self.item_embs = {
            id: self.item_embs[id] if id in self.item_embs
            else self.item_cs_emb for id in ids
        }

    def update_rating_preds(self):
        """
        Measure cosine similarity between user and item embeddings.

        param; user_embs: (dict) user_id: user_emb key-value pairs
        param; item_embs: (dict) item_id: itememb key-value pairs
        """
        # convert to dataframes
        user_embs = pd.DataFrame(self.user_embs)   # (n_users, n_latent)
        item_embs =  pd.DataFrame(self.item_embs)  # (n_items, n_latent)
        # predicted rating matrix
        self.pred_ratings = user_embs.T @ item_embs \
            / np.linalg.norm(item_embs) \
                / np.linalg.norm(user_embs)

    def get_user_support(self, data):
        """
        Get list of items rated by users.
        """
        return data.reset_index().groupby('user_id')\
            ['item_id'].apply(lambda x: list(x))\
                .to_dict()

    def get_item_support(self, data):
        """
        Get list of users who rated items.
        """
        return data.reset_index().groupby('item_id')\
            ['user_id'].apply(lambda x: list(x))\
                .to_dict()

    def get_recommendation(self, user_id):
        """
        Recommend an item for a user.
        """
        rec_item_id = self.pred_ratings.loc[
            user_id, ~self.pred_ratings.columns.isin(self.user_support[user_id])
        ].idxmax()

        return rec_item_id

    def update_embeddings(self, data, user_id, item_id, rating):
        """
        Update user and item embeddings
        """
        # look up user, item datapoint
        x = data.loc[user_id, item_id].to_numpy()
        # get new user embedding
        v_new = self.model.transform(x * rating).ravel()
        # update user and item embeddings
        self.update_user_embedding(v_new, user_id)
        self.update_item_embedding(v_new, item_id)

    def update_user_embedding(self, v_new, user_id):
        """
        param; v_new: transformed user-item embedding of shape (, n_latent)
        """
        # get current user embedding
        v_old = np.array(self.user_embs[user_id])
        # get no. of user ratings, add 1 to count cold-start emb
        n = len(self.user_support[user_id]) + 1 
        # update user embedding
        self.user_embs[user_id] = (1 / (n + 1)) * (n * v_old + v_new)

    def update_item_embedding(self, v_new, item_id):
        """
        param; v_new: transformed user-item embedding of shape (, n_latent)
        """
        # get current user embedding
        v_old = np.array(self.item_embs[item_id])
        # get no. of user ratings, add 1 to count cold-start emb
        n = len(self.item_support[item_id]) + 1 
        # update user embedding
        self.item_embs[item_id] = (1 / (n + 1)) * (n * v_old + v_new)

    # def update_rating_preds(self, user_id):
    #     """
    #     Update rating prediction matrix.
    #     """
    #     user_embs = pd.DataFrame(self.user_embs)
    #     item_embs = pd.DataFrame(self.item_embs)
    #     self.pred_ratings = \
    #         (item_embs @ user_embs[user_id].T)\
    #             .values.reshape(1, -1)

    def update_supports(self, user_id, item_id):
        """
        Update user and item supports
        """
        self.user_support[user_id].append(item_id)
        self.item_support[item_id].append(user_id)

    def init_user_support(self, ids):
        """
        Update support to include unseen users.
        """
        self.user_support = {
            id: self.user_support[id] if id in self.user_support
            else [] for id in ids
        }

    def init_item_support(self, ids):
        """
        Update support to include unseen users.
        """
        self.item_support = {
            id: self.item_support[id] if id in self.item_support
            else [] for id in ids
        }

    def test(self, data, ratings):
        """
        Run replayer experiment.

        param; model: model with transform method for embeddings.
        """
        # get user and item IDs
        user_ids = data.index.get_level_values('user_id')
        item_ids = data.index.get_level_values('item_id')

        # init user and item supports
        self.init_user_support(user_ids.unique())
        self.init_item_support(item_ids.unique())

        # get user and item embeddings
        self.init_user_embeddings(user_ids.unique())
        self.init_item_embeddings(item_ids.unique())

        # update predicted ratings matrix
        self.update_rating_preds()

        # store reccommend item rating in list
        rec_item_ratings = []
        for user_id, item_id, rating in zip(user_ids, item_ids, ratings): 
            # make a movie recommendation
            rec_item_id = self.get_recommendation(user_id)
            # if rec matches historical
            if rec_item_id == item_id:
                # append item rating
                rec_item_ratings.append(rating)
                # update user and item supports
                self.update_supports(user_id, item_id)
                # update user and item embeddings
                # update cold start emb?
                self.update_embeddings(data, user_id, item_id, rating)
                # update rating estimates
                self.update_rating_preds()

        return rec_item_ratings
                    