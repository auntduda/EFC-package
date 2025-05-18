import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report



from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

import sys
import os

sys.path.append('/home/master/Área de Trabalho/energyfc/EFC-package')

from efc import EnergyBasedFlowClassifier
from feature_extraction import extract_ngram_features, extract_partial_features


# essa parte aqui era bom de fazer um scriptzinho que extrai isso direto do site do Mendeley
raw_training = pd.read_csv('/home/master/Área de Trabalho/dataset/dnstun_mendeley/training.csv', names=['ans', 'url'], header=None)
raw_validating = pd.read_csv('/home/master/Área de Trabalho/dataset/dnstun_mendeley/validating.csv', names=['ans', 'url'], header=None)
majestic_million = pd.read_csv('/home/master/Área de Trabalho/dataset/majestic_million.csv', header=0)

# features do dataset -- 12mil/15mil em training sao malignas e 4mil/5mil sao malignas em validating
training = extract_ngram_features(extract_partial_features(raw_training, "url"), "url", majestic_million)

validating = extract_ngram_features(extract_partial_features(raw_validating, "url"), "url", majestic_million)

# print(validating.query('ans == 0').index)

# o dataset de CAIDA so tem fluxos benignos

"""
ate aqui, temos em 'training' e 'validating' um dataframe com as seguintes colunas:

Index(['ans', 'url', 'entropy', 'Nosubd', 'length',
       'length_continuous_integer', 'length_continuous_string',
       'frequency_of_special_character', 'ratio_of_special_character',
       'frequency_of_integer_character', 'ratio_of_integer_character',
       'frequency_of_vowel_character', 'ratio_of_vowel_character',
       'maximum_label_length', 'reputation_value_2',
       'reputation_value_per_ngram_2'],
      dtype='object')

ans                                                                           = target (a resposta usada no treinamento)
as features que eu coletei em formato array([]) que nao sejam ans e nem url   = data (o dado utilizado como referencia)

"""

# print(training)
# print("\n\n\n")
# print(validating)

# training.to_csv("output.csv")

X_train, y_train = np.array(training.iloc[:, 2:16]), training['ans']

X_test, y_test = np.array(validating.iloc[:, 2:16]), validating['ans']

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# fi(ai) - frequencia local
# fij(ai,aj) - frequencia dupla

# nivel de descretizacao foi escolhido pelo valor padrao do modelo e foi reduzido o valor de peso de pseudo contagens pro minimo que roda com 30 niveis de discretizacao
# e o limiar de classificacao foi ajustado conforme a separacao das classes utilizadas
# clf = EnergyBasedFlowClassifier(n_bins=30, cutoff_quantile=0.99, pseudocounts=0.1)

# tabelinho com diferentes hiper parametros pra ver qual deles fica melhor

# com 11 casas decimais ficou show demais a matriz de confusao, mas o f1 score tava memes

clf = EnergyBasedFlowClassifier(n_bins=30, cutoff_quantile=0.99999999999, pseudocounts=0.1) # binario: 90 <= cutoff <= 95
# clf = EnergyBasedFlowClassifier(n_bins=30, cutoff_quantile=0.95) # binario: 90 <= cutoff <= 95
# clf = EnergyBasedFlowClassifier(n_bins=30, cutoff_quantile=0.90) # binario: 90 <= cutoff <= 95

clf.fit(X_train, y_train, base_class=0)

y_pred, y_energies = clf.predict(X_test, return_energies=True)


# vet = y_pred != y_test
# print(vet[vet].index)


# ploting energies
benign = np.where(y_test == 0)[0]
malicious = np.where(y_test == 1)[0]

# print(np.array(benign).size)
# print("\n\n\n")
# print(np.array(malicious).size)

benign_energies = y_energies[benign]
malicious_energies = y_energies[malicious]
cutoff = clf.estimators_[0].cutoff_

bins = np.histogram(y_energies, bins=60)[1]

print(f"accuracy score: {accuracy_score(y_test, y_pred)}")
print(f"f1 score: {f1_score(y_test, y_pred)}")
print(f"classification report: \n{classification_report(y_test, y_pred)}")
# ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
# plt.show()

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


caida_dataset = pd.read_csv('/home/master/Área de Trabalho/dataset/dns-names.l7.20240101.txt', names=['time', 'ip', 'url'], sep='\s+')
# print(caida_dataset)

caida_features = extract_ngram_features(extract_partial_features(caida_dataset, "url"), "url", majestic_million)

caida_repvalue2 = caida_features.query('reputation_value_2 == -1').groupby(['reputation_value_2']).count()
caida_repvaluengram = caida_features.query('reputation_value_per_ngram_2 == -2').groupby(['reputation_value_per_ngram_2']).count()

print(caida_repvalue2)
print(caida_repvaluengram)

# print(caida_features)