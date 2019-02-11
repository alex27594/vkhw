def precision(reals, recs):
    return len(set(recs).intersection(set(reals)))/len(recs)


def average_precision(reals, recs):
    m = len(reals)
    set_reals = set(reals)
    s = sum(precision(reals, recs[:i + 1]) * (1 if recs[i] in set_reals else 0) for i in range(len(recs)))
    return s/m


def mean_average_precision(reals_arr, recs_arr):
    assert len(reals_arr) == len(recs_arr)
    n = len(recs_arr)
    return sum(average_precision(reals_arr[i], recs_arr[i]) for i in range(n))/n