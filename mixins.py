import itertools as it
import numpy as np
import os


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


class SetParameterMixin:
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


class HyperOptimizerMixin:
    def _generate_params_combos(self, params):
        params_names = params.keys()
        params_lists = [params[name] for name in params_names]
        return [dict(zip(params_names, val)) for val in it.product(*params_lists)]

