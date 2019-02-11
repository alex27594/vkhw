
from collections import namedtuple
from sklearn.model_selection import StratifiedKFold

from algos import MySVD, MyPLSA, LDAAdapter
from metrics import mean_average_precision
from mixins import HyperOptimizerMixin
from utils import create_negative_examples, Transformer

TopItem = namedtuple('TopItem', ['combo', 'score'])


class SVDHyperOptimizer(HyperOptimizerMixin):
    def __init__(self, n_splits, K, random_state):
        self.n_splits = n_splits
        self.K = K
        self.random_state = random_state

    def search(self, df, params):
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state)
        params_combos = self._generate_params_combos(params)
        self.top = []
        for combo in params_combos:
            for train_index, test_index in skf.split(X=df, y=df.group):
                df_train = df.iloc[train_index]
                df_test = df.iloc[test_index]

                df_neg = create_negative_examples(df=df, df_train=df_train,
                                                  num_neg_exs=df_train.shape[0], random_state=self.random_state)
                svd = MySVD()
                svd.set_params(**combo)

                svd.fit(df_train, df_neg)

                # transform df_test
                svd_group_encoder_dict = dict(zip(svd.group_encoder.classes_, list(range(len(svd.group_encoder.classes_)))))
                encoded_corpus_test = df_test.groupby("user").agg(lambda x: [svd_group_encoder_dict[item] for item in x]).reset_index()

                encoded_corpus_test['recs'] = svd.predict(encoded_corpus_test.user.values)

                self.top.append(TopItem(combo=combo, score=mean_average_precision(encoded_corpus_test.group.values, encoded_corpus_test.recs.values)))
        self.top = sorted(self.top, key=lambda item: item.score, reverse=True)
        return self.top


class PLSALDAHyperOptimizer(HyperOptimizerMixin):
    def __init__(self, n_splits, K, random_state):
        self.n_splits = n_splits
        self.K = K
        self.random_state = random_state

    def search(self, df, algotype, params):
        assert algotype in ('lda', 'plsa')
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state)
        params_combos = self._generate_params_combos(params)
        self.top = []
        transformer = Transformer()
        transformer.fit(df)
        for combo in params_combos:
            for train_index, test_index in skf.split(X=df, y=df.group):
                df_train = df.iloc[train_index]
                df_test = df.iloc[test_index]

                corpus_train = transformer.transform(df_train)
                corpus_test = transformer.transform(df_test)

                if algotype == 'lda':
                    est = LDAAdapter()
                else:
                    est = MyPLSA()

                est.set_params(**combo)

                est.fit(corpus_train)
                corpus_test['recs'] = est.predict(corpus_test.user.values)

                corpus_test['group'] = corpus_test.group.apply(lambda x: [item[0] for item in x])

                self.top.append(TopItem(combo=combo, score=mean_average_precision(corpus_test.group.values, corpus_test.recs.value)))

        self.top = sorted(self.top, key=lambda item: item.score, reverse=True)
        return self.top