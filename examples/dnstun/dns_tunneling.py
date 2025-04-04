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
data1 = pd.read_csv('/home/master/Área de Trabalho/dataset/dnstun_mendeley/training.csv', names=['ans', 'url'], header=None)
data2 = pd.read_csv('/home/master/Área de Trabalho/dataset/dnstun_mendeley/validating.csv', names=['ans', 'url'], header=None)
majestic_million = pd.read_csv('/home/master/Área de Trabalho/dataset/majestic_million.csv', header=0)
############

# print(data1)
# print(data2)

# tem que concatenar as duas?
data = pd.concat([data1, data2])
########################

# features do dataset
partial = extract_partial_features(data, "url")
ngram = extract_ngram_features(data, "url", majestic_million) # falta o df do majestic million: https://majestic.com/reports/majestic-million


# tem que concatenar as duas?
print(partial.columns)
print("\n\n\n\n")
print(ngram.columns)
##############################3

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
"""

# X, y = np.array(df['entropy', 'Nosubd', 'length',
#        'length_continuous_integer', 'length_continuous_string',
#        'frequency_of_special_character', 'ratio_of_special_character',
#        'frequency_of_integer_character', 'ratio_of_integer_character',
#        'frequency_of_vowel_character', 'ratio_of_vowel_character',
#        'maximum_label_length', 'reputation_value_2',
#        'reputation_value_per_ngram_2']), df['ans']

# print(f"array X: {X}")
# print(f"lista y: {y}")

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, random_state=42, stratify=y, shuffle=True, test_size=0.3
#     # data, random_state=42, shuffle=True, test_size=0.3
# )

# X_train = np.array([[random.randint(0,1e9) for i in range(10)] for j in range(10)]).reshape(-1,1)
# y_train = np.array([random.randint(0,1) for i in range(100)])
# X_test = np.array([[random.randint(0,1e9) for i in range(10)] for j in range(10)]).reshape(-1,1)
# y_test = np.array([random.randint(0,1) for i in range(10)])

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# clf = EnergyBasedFlowClassifier(n_bins=10, cutoff_quantile=0.99)

# clf.fit(X_train, y_train, base_class=0)
# y_pred, y_energies = clf.predict(X_test, return_energies=True)

# # ploting energies
# benign = np.where(y_test == 0)[0]
# malicious = np.where(y_test == 1)[0]

# benign_energies = y_energies[benign]
# malicious_energies = y_energies[malicious]
# cutoff = clf.estimators_[0].cutoff_

# bins = np.histogram(y_energies, bins=60)[1]

# plt.hist(
#     malicious_energies,
#     bins,
#     facecolor="#006680",
#     alpha=0.7,
#     ec="white",
#     linewidth=0.3,
#     label="malicious",
# )
# plt.hist(
#     benign_energies,
#     bins,
#     facecolor="#b3b3b3",
#     alpha=0.7,
#     ec="white",
#     linewidth=0.3,
#     label="benign",
# )
# plt.axvline(cutoff, color="r", linestyle="dashed", linewidth=1)
# plt.legend()

# plt.xlabel("Energy", fontsize=12)
# plt.ylabel("Density", fontsize=12)

# plt.show()