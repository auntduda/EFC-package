import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import sys
import os

sys.path.append('/home/master/Área de Trabalho/energyfc/EFC-package')

from efc import EnergyBasedFlowClassifier
from feature_extraction import extract_ngram_features, extract_partial_features

import random


# essa parte aqui era bom de fazer um scriptzinho que extrai isso direto do site do Mendeley
raw_training = pd.read_csv('/home/master/Área de Trabalho/dataset/dnstun_mendeley/training.csv', names=['ans', 'url'], header=None)
raw_validating = pd.read_csv('/home/master/Área de Trabalho/dataset/dnstun_mendeley/validating.csv', names=['ans', 'url'], header=None)
majestic_million = pd.read_csv('/home/master/Área de Trabalho/dataset/majestic_million.csv', header=0)

# features do dataset
training = extract_ngram_features(extract_partial_features(raw_training, "url"), "url", majestic_million)

validating = extract_ngram_features(extract_partial_features(raw_validating, "url"), "url", majestic_million)

# print(validating.query('ans == 1').shape[0])

"""
ate aqui, temos em 'partial' e 'ngram' um dataframe com as seguintes colunas:

Index(['ans', 'url', 'entropy', 'Nosubd', 'length',
       'length_continuous_integer', 'length_continuous_string',
       'frequency_of_special_character', 'ratio_of_special_character',
       'frequency_of_integer_character', 'ratio_of_integer_character',
       'frequency_of_vowel_character', 'ratio_of_vowel_character',
       'maximum_label_length', 'reputation_value_2',
       'reputation_value_per_ngram_2'],
      dtype='object')

ans                                             = target (a resposta usada no treinamento)
as features que eu coletei em formato array([]) = data (o dado utilizado como referencia)

como se fosse

X, y = np.array(df['entropy', 'Nosubd', 'length',
       'length_continuous_integer', 'length_continuous_string',
       'frequency_of_special_character', 'ratio_of_special_character',
       'frequency_of_integer_character', 'ratio_of_integer_character',
       'frequency_of_vowel_character', 'ratio_of_vowel_character',
       'maximum_label_length', 'reputation_value_2',
       'reputation_value_per_ngram_2']), df['ans']

"""

# print(training)
# print("\n\n\n")
# print(validating)

X_train, y_train = np.array(training.iloc[:, 2:16]), training['ans']

X_test, y_test = np.array(validating.iloc[:, 2:16]), validating['ans']

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

clf = EnergyBasedFlowClassifier(n_bins=10, cutoff_quantile=0.99)

clf.fit(X_train, y_train, base_class=0)

""""
estou tendo um erro de:

Traceback (most recent call last):
  File "/home/master/Área de Trabalho/energyfc/EFC-package/examples/dnstun/dns_tunneling.py", line 84, in <module>
    clf.fit(X_train, y_train, base_class=0)
  File "/home/master/Área de Trabalho/energyfc/EFC-package/efc/_energyclassifier.py", line 121, in fit
    X, y = check_X_y(X, y)
           ^^^^^^^^^^^^^^^
  File "/home/master/Área de Trabalho/energyfc/lib/python3.11/site-packages/sklearn/utils/validation.py", line 1370, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "/home/master/Área de Trabalho/energyfc/lib/python3.11/site-packages/sklearn/utils/validation.py", line 1055, in check_array
    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/master/Área de Trabalho/energyfc/lib/python3.11/site-packages/sklearn/utils/_array_api.py", line 839, in _asarray_with_order
    array = numpy.asarray(array, order=order, dtype=dtype)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string to float: 'q+z8a3wbba.hidemyself.org.'

acho que o EFC ta tentando discretizar tudo do dataframe, mas quando tenta discretizar a url, ele nao consegue por ser uma string

posso so tirar a url do dataframe? desse modo, nao perderei minha chave de identificacao do dataframe? ou o index ja sera o suficiente?
"""

y_pred, y_energies = clf.predict(X_test, return_energies=True)

# print(y_pred)
# print("\n\n\n")
# print(y_energies)

# ploting energies
benign = np.where(y_test == 0)[0]
malicious = np.where(y_test == 1)[0]

# print(benign)
# print("\n\n\n")
# print(np.array(malicious).size)

benign_energies = y_energies[benign]
malicious_energies = y_energies[malicious]
cutoff = clf.estimators_[0].cutoff_

bins = np.histogram(y_energies, bins=60)[1]

plt.hist(
    malicious_energies,
    bins,
    facecolor="#006680",
    alpha=0.7,
    ec="white",
    linewidth=0.3,
    label="malicious",
)
plt.hist(
    benign_energies,
    bins,
    facecolor="#b3b3b3",
    alpha=0.7,
    ec="white",
    linewidth=0.3,
    label="benign",
)
plt.axvline(cutoff, color="r", linestyle="dashed", linewidth=1)
plt.legend()

plt.xlabel("Energy", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.show()