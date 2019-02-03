import numpy as np
import pandas as pd

from collections import Counter
from gensim.models.ldamodel import LdaModel
from sklearn.preprocessing import LabelEncoder


class Transformer:
    def fit(self, df_train):
        self.group_encoder = LabelEncoder()
        self.group_encoder.fit(df_train.group)
        return self

    def transform(self, df_train):
        encoded_df_train = pd.DataFrame(
            {
                "groups": self.group_encoder.transform(df_train.group),
                "users": df_train.user
            }
        )
        encoded_corpus_train = encoded_df_train.groupby("users").agg(lambda x: [(item, 1) for item in x])
        return encoded_corpus_train


class LDAAdapter:
    def __init__(self, num_topics, K, random_state):
        self.num_topics = num_topics
        self.K = K
        self.random_state = random_state
        
    def fit(self, encoded_corpus_train):
        self.encoded_corpus_train = encoded_corpus_train
        self.lda = LdaModel(self.encoded_corpus_train.groups, num_topics=self.num_topics,
                            random_state=self.random_state, distributed=True)
        self.groups_counter = Counter(sum(encoded_corpus_train.groups, []))
        return self

    def predict(self, users_ids):
        terms_topics = self.lda.get_topics()
        recs = []
        for user_id in users_ids:
            if user_id in self.encoded_corpus_train.index:
                user_groups = self.encoded_corpus_train.loc[user_id]["groups"]
                user_topics = np.array([item[1] for item in self.lda.get_document_topics(user_groups)]).reshape((1, -1))
                user_terms = user_topics.dot(terms_topics)
                user_train_groups = [item[0] for item in user_groups]
                user_recs = [item[0] for item in sorted(
                    [item for item in list(enumerate(user_terms.tolist()[0])) if
                     item[0] not in user_train_groups], key=lambda item: item[1], reverse=True)][:self.K]
                recs.append(user_recs)
            else:
                recs.append([item[0][0] for item in self.groups_counter.most_common(self.K)])
        return recs


class MySVD:
    def __init__(self, num_topics, K, random_state):
        self.num_topics = num_topics
        self.K = K
        self.random_state = random_state








