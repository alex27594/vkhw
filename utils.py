import numpy as np
import os
import pandas as pd
import random

from collections import Counter
from gensim.models.ldamodel import LdaModel
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


class LDAAdapter:
    def __init__(self, num_topics, K, random_state):
        self.num_topics = num_topics
        self.K = K
        self.random_state = random_state
        
    def fit(self, corpus_train):
        self.lda = LdaModel(corpus_train.group, num_topics=self.num_topics,
                            random_state=self.random_state)
        self.groups_counter = Counter(sum(corpus_train.group, []))
        self.dict_train = dict(zip(corpus_train.user, corpus_train.group))
        return self

    def predict(self, users_ids):
        terms_topics = self.lda.get_topics()
        recs = []
        for user_id in users_ids:
            if user_id in self.dict_train:
                user_train_groups = self.dict_train[user_id]
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


class MemorizedArrayRecommenderMixin:
    def check_directory(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def make_zero_mem_arr(self, dir_path, name, shape):
        return np.memmap(os.path.join(dir_path, name),
                         dtype=np.float32,
                         mode='w+',
                         shape=shape)

    def make_random_mem_arr(self, dir_path, name, shape):
        mem_arr = self.make_zero_mem_arr(dir_path, name, shape)
        for i in range(shape[0]):
            mem_arr[i, :] = np.random.normal(0.5, 0.1, size=shape[1])
        return mem_arr

    def make_simplex_random_mem_arr(self, dir_path, name, shape):
        mem_arr = self.make_zero_mem_arr(dir_path, name, shape)
        for i in range(shape[0]):
            mem_arr[i, :] = np.random.dirichlet(np.array([1 for j in range(shape[1])]))
        return mem_arr


class ArrayPredictorMixin:
    def _inner_predict(self, user_ids, ut_arr, tg_arr):
        recs = []
        for user_id in user_ids:
            if user_id in self.encoder_dict:
                user_train_groups = self.dict_train[self.encoder_dict[user_id]]
                user_groups_weights = ut_arr[self.encoder_dict[user_id], :].dot(tg_arr)
                user_recs = [item[0] for item in sorted(
                    [item for item in list(enumerate(user_groups_weights)) if
                     item[0] not in user_train_groups], key=lambda item: item[1], reverse=True)][:self.K]
                recs.append(user_recs)
            else:
                recs.append([item[0] for item in self.groups_counter.most_common(self.K)])
        return recs


class PCAExpErrChecker:
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


class MyPCA(MemorizedArrayRecommenderMixin, ArrayPredictorMixin):
    def __init__(self, num_topics, K, max_num_steps, slice_size, exp_coef, tol):
        self.num_topics = num_topics
        self.K = K
        self.max_num_steps = max_num_steps
        self.slice_size = slice_size
        self.exp_coef = exp_coef
        self.tol = tol
        self.arrs_path = 'pca_arrs'

    def fit(self, corpus_train, num_groups):
        self.user_encoder = LabelEncoder()
        self.user_encoder.fit(corpus_train.user)
        num_users = len(self.user_encoder.classes_)
        self.encoder_dict = dict(zip(self.user_encoder.classes_, list(range(num_users))))

        inner_corpus_train = pd.DataFrame(
            {
                'user': self.user_encoder.transform(corpus_train.user.values),
                'group': corpus_train.group.apply(lambda x: [item[0] for item in x])
            }
        )

        self.check_directory(self.arrs_path)
        self.put_arr = self.make_simplex_random_mem_arr(
            dir_path=self.arrs_path,
            name='put_arr.dat',
            shape=(num_users, self.num_topics)
        )
        self.ptg_arr = self.make_simplex_random_mem_arr(
            dir_path=self.arrs_path,
            name='ptg_arr.dat',
            shape=(self.num_topics, num_groups)
        )
        self.nut_arr = self.make_zero_mem_arr(
            dir_path=self.arrs_path,
            name='nut_arr.dat',
            shape=(num_users, self.num_topics)
        )
        self.ntg_arr = self.make_zero_mem_arr(
            dir_path=self.arrs_path,
            name='ntg_arr.dat',
            shape=(self.num_topics, num_groups)
        )
        self.nt_arr = np.zeros(self.num_topics)
        self.nu_arr = np.zeros(num_users)
        put_exp_err_checker = PCAExpErrChecker(slice_size=self.slice_size, exp_coef=self.exp_coef, tol=self.tol)
        ptg_exp_err_checker = PCAExpErrChecker(slice_size=self.slice_size, exp_coef=self.exp_coef, tol=self.tol)


        counter = 0
        for i in range(self.max_num_steps):
            print(counter)
            counter += 1
            print('put_arr', self.put_arr)
            print('ptg_arr', self.ptg_arr)
            for u in range(num_users):
                row = [item[0] for item in inner_corpus_train[inner_corpus_train.user == u].group]
                for g in row:
                    nug = 1
                    pug = self.put_arr[u, :].dot(self.ptg_arr[:, g])
                    for t in range(self.num_topics):
                        pugt = (self.put_arr[u, t] * self.ptg_arr[t, g]) / pug
                        self.ntg_arr[t, g] += nug * pugt
                        self.nut_arr[u, t] += nug * pugt
                        self.nt_arr[t] += nug * pugt
                        self.nu_arr[u] += nug * pugt
            for t in range(self.num_topics):
                self.ptg_arr[t, :] = self.ntg_arr[t, :] / self.nt_arr[t]
                self.put_arr[:, t] = self.nut_arr[:, t] / self.nu_arr

            if put_exp_err_checker.check(self.put_arr) and ptg_exp_err_checker.check(self.ptg_arr):
                break

            self.ntg_arr[:, :] = 0
            self.nut_arr[:, :] = 0
            self.nt_arr[:] = 0
            self.nu_arr[:] = 0

        self.dict_train = dict(zip(inner_corpus_train.user, inner_corpus_train.group))
        self.groups_counter = Counter(sum(inner_corpus_train.group, []))
        return self

    def predict(self, user_ids):
        return self._inner_predict(user_ids, self.put_arr, self.ptg_arr)


def create_negative_examples(df, df_train, num_neg_exs, random_state):
    np.random.seed(random_state)
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


class MySVD(MemorizedArrayRecommenderMixin, ArrayPredictorMixin):
    def __init__(self, num_topics, K, max_num_steps, step_size, exp_coef, tol, random_state):
        self.num_topics = num_topics
        self.K = K
        self.max_num_steps = max_num_steps
        self.step_size = step_size
        self.exp_coef = exp_coef
        self.tol = tol
        self.random_state = random_state
        self.arrs_path = 'svd_arrs'
        random.seed(random_state)

    def _train_gen(self, df_train, df_neg_exs):
        while True:
            is_positive_ex = random.choice((0, 1))
            if is_positive_ex:
                row_id = random.randint(0, df_train.shape[0] - 1)
                yield df_train.iloc[row_id].values, 1
            else:
                row_id = random.randint(0, df_neg_exs.shape[0] - 1)
                yield df_neg_exs.iloc[row_id].values, 0

    def fit(self, df_train, df_neg_exs):
        self.user_encoder = LabelEncoder()
        self.user_encoder.fit(df_train.user)

        self.group_encoder = LabelEncoder()
        self.group_encoder.fit(df_train.group)
        num_users = len(self.user_encoder.classes_)
        num_groups = len(self.group_encoder.classes_)

        self.encoder_dict = dict(zip(self.user_encoder.classes_, list(range(num_users))))

        inner_df_train = pd.DataFrame(
            {
                'user': self.user_encoder.transform(df_train.user.values),
                'group': self.group_encoder.transform(df_train.group.values)
            }
        )

        inner_df_neg_exs = pd.DataFrame(
            {
                'user': self.user_encoder.transform(df_neg_exs.user.values),
                'group': self.group_encoder.transform(df_neg_exs.group.values)
            }
        )

        self.check_directory(self.arrs_path)
        self.ut_arr = self.make_random_mem_arr(
            dir_path=self.arrs_path,
            name='ut_arr.dat',
            shape=(num_users, self.num_topics)
        )
        self.tg_arr = self.make_random_mem_arr(
            dir_path=self.arrs_path,
            name='tg_arr.dat',
            shape=(self.num_topics, num_groups)
        )

        ex_gen = self._train_gen(inner_df_train, inner_df_neg_exs)
        exp_err_checker = SVDExpErrChecker(self.exp_coef, self.tol)
        for i in range(self.max_num_steps):
            print(i)
            coords, val = next(ex_gen)
            prev_u = self.ut_arr[coords[0], :]
            prev_g = self.tg_arr[:, coords[1]]
            err = val - prev_u.dot(prev_g)
            self.ut_arr[coords[0], :] = prev_u + self.step_size * err * prev_g
            self.tg_arr[:, coords[1]] = prev_g + self.step_size * err * prev_u
            if exp_err_checker.check(err):
                break

        corpus_train = inner_df_train.groupby("user").agg(lambda x: [item for item in x]).reset_index()
        self.dict_train = dict(zip(corpus_train.user, corpus_train.group))
        self.groups_counter = Counter(sum(corpus_train.group, []))
        return self

    def predict(self, user_ids):
        return self._inner_predict(user_ids, self.ut_arr, self.tg_arr)














