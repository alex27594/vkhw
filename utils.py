import itertools as it
import numpy as np
import pandas as pd
import random

from collections import Counter
from gensim.models.ldamodel import LdaModel
from sklearn.preprocessing import LabelEncoder


class Transformer:
    def fit(self, df_train):
        self.group_encoder = LabelEncoder()
        self.group_encoder.fit(df_train.group)
        self.user_encoder = LabelEncoder()
        self.user_encoder.fit(df_train.user)
        return self

    def transform(self, df_train):
        encoded_df_train = pd.DataFrame(
            {
                "group": self.group_encoder.transform(df_train.group),
                "user": self.user_encoder.transform(df_train.user)
            }
        )
        encoded_corpus_train = encoded_df_train.groupby("user").agg(lambda x: [(item, 1) for item in x]).reset_index()
        return encoded_corpus_train


class LDAAdapter:
    def __init__(self, num_topics, K, random_state):
        self.num_topics = num_topics
        self.K = K
        self.random_state = random_state
        
    def fit(self, encoded_corpus_train):
        self.lda = LdaModel(encoded_corpus_train.group, num_topics=self.num_topics,
                            random_state=self.random_state, distributed=True)
        self.groups_counter = Counter(sum(encoded_corpus_train.group, []))
        self.encoded_dict_train = dict(zip(encoded_corpus_train.user, encoded_corpus_train.group))
        return self

    def predict(self, users_ids):
        terms_topics = self.lda.get_topics()
        recs = []
        for user_id in users_ids:
            if user_id in self.encoded_dict_train:
                user_train_groups = self.encoded_dict_train[user_id]
                user_topics = np.array([item[1] for item in self.lda.get_document_topics(user_train_groups)]).reshape((1, -1))
                user_groups_weights = user_topics.dot(terms_topics)
                user_train_groups = [item[0] for item in user_train_groups]
                user_recs = [item[0] for item in sorted(
                    [item for item in list(enumerate(user_groups_weights.tolist()[0])) if
                     item[0] not in user_train_groups], key=lambda item: item[1], reverse=True)][:self.K]
                recs.append(user_recs)
            else:
                recs.append([item[0][0] for item in self.groups_counter.most_common(self.K)])
        return recs


def create_negative_examples(df, df_train, num_neg_exs):
    all_groups = df.groups.unique()
    all_train_users = df_train.users.unique()
    assert num_neg_exs < all_groups * all_train_users - df.shape[0]
    edges_set = set(tuple(item) for item in df.values)
    neg_exs = []
    while len(neg_exs) < num_neg_exs:
        ex = (random.choice(all_groups), random.choice(all_train_users))
        if ex not in edges_set:
            neg_exs.append(ex)
    return neg_exs


class MySVD:
    def __init__(self, num_topics, K, num_steps, step_size, random_state):
        random.seed(random_state)
        self.num_topics = num_topics
        self.K = K
        self.random_state = random_state
        self.num_steps = num_steps
        self.step_size = step_size

    def train_gen(self, df_train, df_neg_exs):
        is_positive_ex = random.choice([0, 1])
        while True:
            if is_positive_ex:
                row_id = random.randint(0, df_train.shape[0] - 1)
                yield np.flip(df_train.iloc[row_id].values, axis=0), 1
            else:
                row_id = random.randint(0, df_neg_exs.shape[0] - 1)
                yield np.flip(df_neg_exs.iloc[row_id].values, axis=0), 0

    def fit(self, df_train, df_neg_exs):
        num_groups = len(df_train.groups.unique())
        num_users = len(df_train.users.unique())
        self.ut_arr = np.memmap('ut_arr.dat',
                                dtype=np.float32,
                                mode='w+',
                                shape=(num_users, self.num_topics))
        self.tg_arr = np.memmap('tg_arr.dat',
                               dtype=np.float32,
                               mode='w+',
                               shape=(self.num_topics, num_groups))
        ex_gen = self.train_gen(df_train, df_neg_exs)
        for i in range(self.num_steps):
            coords, val = next(ex_gen)
            prev_u = self.ut_arr[coords[0], :]
            prev_g = self.tg_arr[:, coords[1]]
            err = val - prev_u.dot(prev_g)
            self.ut_arr[coords[0], :] = prev_u + self.step_size * err * prev_g
            self.tg_arr[:, coords[1]] = prev_g + self.step_size * err * prev_u
        corpus_train = df_train.groupby("user").agg(lambda x: [item for item in x]).reset_index()
        self.dict_train = dict(zip(corpus_train.user, corpus_train.groups))
        self.groups_counter = Counter(sum(corpus_train.groups, []))
        return self

    def predict(self, user_ids):
        recs = []
        for user_id in user_ids:
            if user_id in self.dict_train:
                user_train_groups = self.dict_train[user_id]
                user_groups_weights = self.ut_arr[user_id, :].dot(self.tg_arr)
                user_recs = [item[0] for item in sorted(
                    [item for item in list(enumerate(user_groups_weights)) if
                     item[0] not in user_train_groups], key=lambda item: item[1], reverse=True)][:self.K]
                recs.append(user_recs)
            else:
                recs.append([item[0] for item in self.groups_counter.most_common(self.K)])
        return recs














