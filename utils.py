import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder


class Transformer:
    def fit(self, df):
        self.group_encoder = LabelEncoder()
        self.group_encoder.fit(df.group)
        return self

    def transform(self, df):
        encoded_df_train = pd.DataFrame(
            {
                "group": self.group_encoder.transform(df.group.values),
                "user": df.user.values
            }
        )
        encoded_corpus_train = encoded_df_train.groupby("user").agg(lambda x: [(item, 1) for item in x]).reset_index()
        return encoded_corpus_train


class PLSAExpErrChecker:
    def __init__(self, slice_size, exp_coef, tol):
        self.slice_size = slice_size
        self.exp_coef = exp_coef
        self.tol = tol
        self.prev_slice_arr = None
        self.prev_slice_ids = None
        self.exp_err = 0

    def check(self, arr):
        if self.prev_slice_arr is None:
            self.exp_err = 1
        else:
            cur_slice_arr = arr[self.prev_slice_ids[0], self.prev_slice_ids[1]]
            self.exp_err = self.exp_coef * self.exp_err + (1 - self.exp_coef) * np.linalg.norm(self.prev_slice_arr - cur_slice_arr)

        self.prev_slice_ids = (np.random.randint(low=0, high=arr.shape[0], size=self.slice_size),
                               np.random.randint(low=0, high=arr.shape[1], size=self.slice_size))
        self.prev_slice_arr = arr[self.prev_slice_ids[0], self.prev_slice_ids[1]]
        return self.exp_err < self.tol


def create_negative_examples(df, df_train, num_neg_exs, random_state):
    random.seed(random_state)
    all_groups = df.group.unique()
    all_train_users = df_train.user.unique()
    assert num_neg_exs <= len(all_groups) * len(all_train_users) - df.shape[0]
    edges_set = set(tuple(item) for item in df.values)
    neg_edges_set = set()
    while len(neg_edges_set) < num_neg_exs:
        ex = (random.choice(all_groups), random.choice(all_train_users))
        if ex not in edges_set and ex not in neg_edges_set:
            neg_edges_set.add(ex)
    neg_edges_list = list(neg_edges_set)
    df_neg_exs = pd.DataFrame({
        'group': [item[0] for item in neg_edges_list],
        'user': [item[1] for item in neg_edges_list]
    })
    return df_neg_exs


class SVDExpErrChecker:
    def __init__(self, exp_coef, tol):
        assert (exp_coef > 0) and (exp_coef < 1)
        self.exp_coef = exp_coef
        self.tol = tol
        self.exp_err = 1

    def check(self, err):
            self.exp_err = self.exp_coef * self.exp_err + (1 - self.exp_coef) * err
            return self.exp_err < self.tol

