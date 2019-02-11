# vkhw

## Сбор данных
collect.php

## Рекомендательные алгоритмы
algos.py
* **LDAAdapter** - реализация рекомендательных систем на основе библиотечного LDA (библиотек gensim)
* **MySVD** - самодельный алгоритм по созданию прогнозов с помощью SVD разложения, 
основанном на стохастическом градиенте
* **MyPLSA** - самодельный алгоритм по созданию прогнозов с помощью подхода PLSA

## Метрики
metrics.py

## Алгоритмы для gridsearch-подбора гиперпараметров
grid_search.py
* **SVDHyperOptimizer** - gridsearch подбор параметров для MySVD
* **PLSALDAHyperOptimizer** - gridsearch подбор параметров для MyPLSA и LDAAdapter
 
## Вспомогательный код
utils.py
mixins.py

## Notebook автоматического подбора параметров с помощью grid search
hyper_opt.ipynb

## Notebook сравнения метрик качества алгоритмов
compare_algos.ipynb
