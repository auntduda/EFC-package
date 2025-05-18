import math
from collections import Counter
from math import log
from nltk import ngrams


def shannon(fqnd):
  counts = Counter(fqnd)
  frequencies = ((i / len(fqnd)) for i in counts.values())
  return - sum(f * log(f, 2) for f in frequencies)

def nosubd(fqdn):
   return fqdn.count('.')-1

def maximum_label_length(fqdn):
    maximum_label_len, curr_len = 0, 0
    for c in fqdn:
        if c != '.':
            curr_len += 1
            if curr_len > maximum_label_len:
                maximum_label_len = curr_len
        else:
            curr_len = 0
    return maximum_label_len

def length(fqdn):
    return len(fqdn)

def length_continuous_integer(fqdn):
    max_continous_ints, ints = 0, 0
    for c in fqdn:
        if c.isdigit():
            ints += 1
            if ints > max_continous_ints:
               max_continous_ints = ints
        else:
            ints = 0
    return max_continous_ints

def length_continuous_string(fqdn):
    max_continous_str, strs = 0, 0
    for c in fqdn:
        if c.isalpha():
            strs += 1
            if strs > max_continous_str:
               max_continous_str = strs
        else:
            strs = 0
    return max_continous_str

def frequency_of_special_character(fqdn):
    # Note that '.' is not consireded as a special character!
    special_characters = 0
    for c in fqdn:
        if c.isdigit() == False and c.isalpha() == False and c != '.':
            special_characters += 1
    return special_characters

def ratio_of_special_character(fqdn):
    # Note that '.' is not consireded as a special character!
    special_characters, sz = 0, 0
    for c in fqdn:
        if c.isdigit() == False and c.isalpha() == False and c != '.':
            special_characters += 1
        sz += 1
    return special_characters/sz

def frequency_of_integer_character(fqdn):
    integer_characters = 0
    for c in fqdn:
        if c.isdigit():
            integer_characters += 1
    return integer_characters

def ratio_of_integer_character(fqdn):
    integer_characters, sz = 0, 0
    for c in fqdn:
        if c.isdigit():
            integer_characters += 1
        sz += 1
    return round(integer_characters/sz, 5)

def frequency_of_vowel_character(fqdn):
    vowel_characters = 0
    for c in fqdn.lower():
        if c in 'aeiou':
            vowel_characters += 1
    return vowel_characters

def ratio_of_vowel_character(fqdn):
    vowel_characters, sz = 0, 0
    for c in fqdn.lower():
        if c in 'aeiou':
            vowel_characters += 1
        sz += 1
    return round(vowel_characters/sz, 5)

# ---------------------------------------------------------------------------
# N-gram based features (Reputation value)

def remove_tld(domain, tld):
    TLD_LIST = ['.edu.', '.gov.', '.com.', '.co.', '.org.', '.ac.', '.ne.', '.net.', '.mil.', '.int.', '.go.jp', '.or.jp', '.lg.jp', '.ad.jp', '.gr.jp']
    
    for aux_tld in TLD_LIST:
        find = domain.find(aux_tld)
        if find != -1:
            return domain[:find]
    return domain[:-(len(tld)+1)]

def remove_unknow_tld(domain):
    find = domain.rfind('.')

    if find == -1:
        print(f'Não foi possivel extrair o TLD do dominio {domain}')
        return None

    return domain[:find]

def remove_unknow_tld_and_sld(domain):
    find1 = domain.rfind('.')
    find2 = domain[:find1].rfind('.')

    if find2 == -1:
        print(f'Não foi possivel extrair o TLD e SLD do dominio {domain}')
        return None

    return domain[:find2]

def get_ngram_frequencies(df_majestic, n_list=[2,3,4,5,6,7]):

    domains_without_tld_list = df_majestic.apply(lambda df: remove_tld(df['Domain'], df['TLD']), axis=1)

    ngram_frequencies = {}
    for n in n_list:
        ngram_frequencies[f"{n}-grams"] = {}
    
    for domain_without_tld in domains_without_tld_list:
        label_list = domain_without_tld.split('.')
        for label in label_list:
            for n in n_list:
                curr_ngrams = ngrams(label, n)
                for ngram in curr_ngrams:
                    string_ngram = ''.join(ngram)
                    if string_ngram in ngram_frequencies[f"{n}-grams"]:
                        ngram_frequencies[f"{n}-grams"][string_ngram] += 1
                    else:
                        ngram_frequencies[f"{n}-grams"][string_ngram] = 1
    
    return ngram_frequencies

def weight_2_ngram(ngram_str, dict_ngram_frequencies):
    ngram_key = str(len(ngram_str)) + '-grams'
    if ngram_str not in dict_ngram_frequencies[ngram_key]:
        return 0
    return math.log2(dict_ngram_frequencies[ngram_key][ngram_str] * len(ngram_str))

def reputation_value(domain, dict_ngram_frequencies, weight_ngram_func):
    domain = remove_unknow_tld(domain)

    if domain == '':
        return None
    elif domain == None:
        return -1

    reputation_value = 0
    n_list = [int(st[0]) for st in dict_ngram_frequencies.keys()]
    label_list = domain.split('.')
    for label in label_list:
        for n in n_list:
            curr_ngrams = ngrams(label, n)
            for ngram in curr_ngrams:
                string_ngram = ''.join(ngram)
                if string_ngram in dict_ngram_frequencies[f"{n}-grams"]:
                    reputation_value += weight_ngram_func(string_ngram, dict_ngram_frequencies)
                else:
                    reputation_value += weight_ngram_func(string_ngram, dict_ngram_frequencies)
    return reputation_value

def reputation_value_per_ngram(domain, dict_ngram_frequencies, weight_ngram_func):
    domain = remove_unknow_tld(domain)

    if domain == None:
        return -2

    reputation_value = 0
    ngram_count = 0
    n_list = [int(st[0]) for st in dict_ngram_frequencies.keys()]
    label_list = domain.split('.')
    for label in label_list:
        for n in n_list:
            curr_ngrams = ngrams(label, n)
            for ngram in curr_ngrams:
                ngram_count += 1
                string_ngram = ''.join(ngram)
                if string_ngram in dict_ngram_frequencies[f"{n}-grams"]:
                    reputation_value += weight_ngram_func(string_ngram, dict_ngram_frequencies)
                else:
                    reputation_value += weight_ngram_func(string_ngram, dict_ngram_frequencies)
    if ngram_count == 0:
        return None
    return reputation_value/ngram_count
# ---------------------------------------------------------------------------

# df: Pandas DataFrame to extract features
# qcol: DNS Query column name from the DataFrame
def extract_partial_features(df, qcol):
    df[qcol] = df[qcol].astype(str)
    df[qcol] = df[qcol].apply(lambda d: d.lower())

    df['entropy'] = df[qcol].apply(shannon)
    df['Nosubd'] = df[qcol].apply(nosubd)
    df['length'] = df[qcol].apply(length)
    df['length_continuous_integer'] = df[qcol].apply(length_continuous_integer)
    df['length_continuous_string'] = df[qcol].apply(length_continuous_string)
    df['frequency_of_special_character'] = df[qcol].apply(frequency_of_special_character)
    df['ratio_of_special_character'] = df[qcol].apply(ratio_of_special_character)
    df['frequency_of_integer_character'] = df[qcol].apply(frequency_of_integer_character)
    df['ratio_of_integer_character'] = df[qcol].apply(ratio_of_integer_character)
    df['frequency_of_vowel_character'] = df[qcol].apply(frequency_of_vowel_character)
    df['ratio_of_vowel_character'] = df[qcol].apply(ratio_of_vowel_character)
    df['maximum_label_length'] = df[qcol].apply(maximum_label_length)

    return df

# df: Pandas DataFrame
# qcol: DNS Query column name from the DataFrame
# df_majestic_raw: Auxiliar dataframe to extract a reference distribution of n-grams. In our case, we used Majestic Million domains
def extract_ngram_features(df, qcol, df_majestic_raw, top_n_domains=100000, n_list=[2,3,4,5,6,7]):
    df_majestic = df_majestic_raw[:top_n_domains][['Domain', 'TLD']]
    majestic_100k_ngram_frequencies = get_ngram_frequencies(df_majestic, n_list)

    df[qcol] = df[qcol].astype(str)
    df[qcol] = df[qcol].apply(lambda d: d.lower())

    df['reputation_value_2'] = df.apply(lambda df_aux: reputation_value(df_aux[qcol], majestic_100k_ngram_frequencies, weight_2_ngram), axis=1)
    df['reputation_value_per_ngram_2'] = df.apply(lambda df_aux: reputation_value_per_ngram(df_aux[qcol], majestic_100k_ngram_frequencies, weight_2_ngram), axis=1)
    
    return df
