#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:50:31 2021

@author: raphaelstedler

This is an evaluation script for the task of spelling correction of short German queries written in German.
   This version uses phonetic information (Cologne Phonetics) in the corretion process.
   The following paramters can be set:
   Line 190: The maximum edit distance can be set.
   Line 212: The weght of the phonetic similarity can be set.
   Line 213: The weight of the string similarity can be set.
   The script returns Recall, Recall@k, Precision, F-score, Mean reciprocal rank and maximum found words.
   The construction of the try takes 8-9 minutes, so please be patient.
"""

import numpy as np
import string
import datrie
import datetime
from abydos.phonetic import Koelner
import textdistance
import pandas as pd
from tqdm import tqdm
import itertools
import os


'''instantiate Cologne Phonetics'''
koe = Koelner()

'''relative path'''
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

'''error list'''
df_Fehlerliste_final = pd.read_csv('finale_fehlerliste_reduziert.csv', sep=';')
df_Fehlerliste_final = df_Fehlerliste_final[['Error_Word', 'Correct_Word']]

col_comb = df_Fehlerliste_final[['Correct_Word', 'Error_Word']].values.tolist()


'''all to lower case'''
col_comb2 = []
for i in range(len(col_comb)):
    col_comb2.append([col_comb[i][0].lower(), col_comb[i][1].lower()])
col_comb = col_comb2

'''deduplicate'''
k = sorted(col_comb)
dedup = [k[i] for i in range(len(k)) if i == 0 or k[i] != k[i-1]]
col_comb = dedup
del dedup

'''error list'''
col_comb_kp = []

for i in range(len(col_comb)):
    col_comb_kp.append([col_comb[i][0], col_comb[i][1], koe.encode_alpha(col_comb[i][1]).lower()])


'''col_comb_dict for kp'''
col_comb_dict = {}
for i in range(len(col_comb_kp)):
    key = col_comb_kp[i][1]
    value = col_comb_kp[i][0]
    try:
        col_comb_dict[key] += value
    except KeyError:
        col_comb_dict[key] = value

DICTIONARY = "kölnalpha_stacked.txt"
filename = "kölnalpha_origwords_lower.txt"

emty_dict = {}
with open(filename) as f:
  for line in f:
      data = line.strip().split()
      key = data[0].lower()
      value = data[1:]
      try:
          emty_dict[key] += data[1:]
      except KeyError:
          emty_dict[key] = data[1:]

strings_set = set()

for line in open('kölnalpha_stacked.txt').readlines():
 	strings_set |= set([line.split('\n')[0]])

strings_list = list(strings_set)

without_empty_strings = []
for i in strings_list:
    if (i != ""):
        without_empty_strings.append(i)
        
strings_set = set(without_empty_strings)
 
'''filter capital letters off the set'''
strings_set_lower = {s.lower() for s in strings_set}

'''class Damerau-Levenshtein Trie'''
class DamerauLevenshteinTrie(object):
  def __init__(self, strings, dictionary=None, max_calc_flag=False):
      # optional optimization flag based on string lengths
      self.max_calc_flag = max_calc_flag
      # collecting all chars etc
      if dictionary is None:
        self.chars, self.initial_chars, self.strings = self.get_chars(strings)
        # building the trie
        self.dictionary = self.get_dictionary(self.strings, self.chars)
      else:
        # trie was supplied as input
        self.dictionary = dictionary
        self.chars = set()
        self.initial_chars = set()
        for s in self.dictionary:
          self.chars |= set(s)
          self.initial_chars |= set(s[0])
          
  def get_chars(self, strings):   
      chars = set()
      initial_chars = set()
      robust_strings = set()
      for s in set(strings):
        if len(s) < 1:
          continue
        chars |= set(s)
        initial_chars |= set(s[0])
        robust_strings |= set([s])
      return ''.join(chars), initial_chars, robust_strings
    
  def get_dictionary(self, strings, chars):
      strings_trie = datrie.Trie(string.ascii_lowercase)
      for s in strings:
        strings_trie[s] = 0
      return strings_trie
  
  # a way to get all possible next characters given some prefix
  def edges(self, dictionary, prefix, chars, max_dist=False):
      if max_dist:
        return {l for l in chars if dictionary.has_keys_with_prefix(prefix + l) and max([len(k) for k in dictionary.keys(prefix + l)]) > max_dist}
      return {l for l in chars if dictionary.has_keys_with_prefix(prefix + l)}

  def search(self, string, max_cost):
      current_row = range(len(string) + 1)
      results = set()

      for char in self.initial_chars:
        self.search_recursive(char, char, None, string, current_row, None, results, max_cost)
      return results

  def search_recursive(self, prefix, char, prev_char, string, previous_row, pre_previous_row, results, max_cost):
      columns = len(string) + 1
      current_row = [previous_row[0] + 1]

      for column in range(1, columns):
        insert_cost = previous_row[column] + 1
        delete_cost = current_row[column - 1] + 1

        if string[column - 1] != char:
          replace_cost = previous_row[column - 1] + 1
        else:                
          replace_cost = previous_row[column - 1]

        current_row.append(min(insert_cost, delete_cost, replace_cost))
        if prev_char and column - 1 > 0 and char == string[column-2] and prev_char == string[column-1] and string[column-1] != char:
            current_row[column] = min(current_row[column], pre_previous_row[column-2] + 1)

      if current_row[-1] <= max_cost and self.dictionary.get(prefix) != None:
        results.add((prefix, current_row[-1]))

      if min(current_row) <= max_cost:
        prev_char = char
        max_calc = max_cost if self.max_calc_flag else None
        for char in self.edges(self.dictionary, prefix, self.chars, max_calc):
          self.search_recursive(prefix + char, char, prev_char, string, current_row, previous_row, results, max_cost)
          
'''wanted_words kp'''
wanted_words = col_comb_kp

'''initiate the trie using the given data'''
start = datetime.datetime.now()
dlt = DamerauLevenshteinTrie(strings_set_lower)
print (datetime.datetime.now() - start)

start = datetime.datetime.now()

'''set maximum edit distance'''
maximum_distance = 1

distances_list = []

for w in tqdm(range(len(wanted_words))):
  distances_list.append([wanted_words[w], dlt.search(wanted_words[w][2], maximum_distance)])


lili = []
for i in range(len(distances_list)):
    lili.append([ distances_list[i][0] ,list(distances_list[i][1])])


'''comparison of original string with dictionary entries and calculation of normalized Damerau-Levenshtein distance'''

li4 = []
for i in tqdm(range(len(lili))):
    try:
        for j in range(len(lili[i][1])):
            if lili[i][1][j][0] in emty_dict:
                for k in range(len(emty_dict[lili[i][1][j][0]])):
                    '''setting of weights for normalized Damerau-Levenshtein distance'''
                    PHONETIC_WEIGHT = 1
                    STRING_SIM_WEIGHT = 3
                    SUM_WEIGTHS = PHONETIC_WEIGHT + STRING_SIM_WEIGHT
                    li4.append([lili[i][0][1],((PHONETIC_WEIGHT*textdistance.damerau_levenshtein.normalized_similarity(lili[i][0][2], lili[i][1][j][0])+(STRING_SIM_WEIGHT*textdistance.damerau_levenshtein.normalized_similarity(lili[i][0][1], emty_dict[lili[i][1][j][0]][k])))/SUM_WEIGTHS), emty_dict[lili[i][1][j][0]][k]])
    except IndexError:
        pass


        
'''direct comparison of original string with dictionary entries'''

dict_items = {}

for i in range(len(li4)):
    try:
        dict_items[li4[i][0]] += [li4[i][1:]]
    except KeyError:
        dict_items[li4[i][0]] = [li4[i][1:]]
        
def Sort(sub_li): 
    sub_li.sort(key = lambda x: x[0], reverse=True) 
    return sub_li 

for i in dict_items.keys():
    Sort(dict_items[i])
    
    


'''counting how many hits there actually are'''

final_li=[]
for i in range(len(col_comb_kp)):
    try:
        if col_comb_kp[i][1] in dict_items:
            for j in range(len(dict_items[col_comb_kp[i][1]])):
            # for j in range(20):
                if col_comb_kp[i][0] == dict_items[col_comb_kp[i][1]][j][1]:
                    final_li.append([col_comb_kp[i][1], dict_items[col_comb_kp[i][1]][j], j])
    except KeyError:
        pass

print (datetime.datetime.now() - start)


final_li.sort()
final_li = list(k for k,_ in itertools.groupby(final_li))


count = 0
for i in range(len(final_li)):
    if final_li[i][2] == 0:
        count+=1
recallat11 = count
recallat1 = count/len(col_comb)

count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 1:
        count+=1
recallat22 = count
recallat2 = count/len(col_comb)

        
count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 2:
        count+=1
recallat3 = count/len(col_comb)
recallat33 = count

count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 3:
        count+=1
recallat4 = count/len(col_comb)
recallat44 = count

count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 4:
        count+=1
recallat5 = count/len(col_comb)
recallat55 = count
        
count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 5:
        count+=1
recallat6 = count/len(col_comb)
recallat66 = count

count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 6:
        count+=1
recallat7 = count/len(col_comb)
recallat77 = count

count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 7:
        count+=1
recallat8 = count/len(col_comb)
recallat88 = count

count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 8:
        count+=1
recallat9 = count/len(col_comb)
recallat99 = count

count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 9:
        count+=1
recallat10 = count/len(col_comb)
recallat100 = count
        
count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 19:
        count+=1
recallat20 = count/len(col_comb)
recallat200 = count
        
count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 29:
        count+=1
recallat30 = count/len(col_comb)
recallat300 = count
        
count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 39:
        count+=1
recallat40 = count/len(col_comb)
recallat400 = count

count = 0
for i in range(len(final_li)):
    if final_li[i][2] <= 49:
        count+=1
recallat50 = count/len(col_comb)
recallat500 = count


# '''check, which words in the suggestion list are not among the top k candidates'''
# count = 0
# k = 5
# for i in range(len(final_li)):
#     if final_li[i][2] >k-1:
#         print(final_li[i])



final_li_dict ={}
for i in range(len(final_li)):
    key = final_li[i][0]
    value = final_li[i][1]
    final_li_dict[key] = value

recall_li = []
for i in range(len(li4)):
    if li4[i][0] in final_li_dict:
        recall_li.append(li4[i])
        
    
l=[]
[l.extend([v]) for k,v in dict_items.items()]
recall_count =0
for i in range(len(l)):
    recall_count += len(l[i])

'''Precision'''
precision = len(final_li)/len(li4)

'''Recall'''
recall = len(final_li)/len(col_comb)

'''F-Score'''
fscore = (2*precision*recall)/(precision+recall)

'''max words found'''
max_found = len(final_li)

print('Metric\t\t\t Found words\t\t\t Scores')
print('Recall@1:\t\t',recallat11 ,'\t\t\t\t',recallat1)
print('Recall@2:\t\t' ,recallat22 ,'\t\t\t\t',recallat2)
print('Recall@3:\t\t' ,recallat33,'\t\t\t\t' ,recallat3)
print('Recall@4:\t\t' ,recallat44,'\t\t\t\t' ,recallat4)
print('Recall@5:\t\t' ,recallat55,'\t\t\t\t' ,recallat5)
print('Recall@6:\t\t',recallat66,'\t\t\t\t' ,recallat6)
print('Recall@7:\t\t',recallat77,'\t\t\t\t' ,recallat7)
print('Recall@8:\t\t',recallat88,'\t\t\t\t' ,recallat8)
print('Recall@9:\t\t',recallat99,'\t\t\t\t' ,recallat9)
print('Recall@10:\t\t',recallat100,'\t\t\t\t' ,recallat10)
print('Recall@20:\t\t', recallat200 ,'\t\t\t\t',recallat20)
print('Recall@30:\t\t',recallat300 ,'\t\t\t\t',recallat30)
print('Recall@40:\t\t',recallat400,'\t\t\t\t' ,recallat40)
print('Recall@50:\t\t',recallat500,'\t\t\t\t' ,recallat50)
print('max_words_found: ','\t\t\t\t\t', ((max_found/len(col_comb))*100),'%')
print('Precision: ','\t\t\t\t\t\t\t' ,precision)
print('Recall_gesamt: ','\t\t\t\t\t\t',recall)
print('F-score: ','\t\t\t\t\t\t\t',fscore)


'''Mean reciprocal rank'''

mrr_list=[]
count = 5000
for i in range(len(final_li)):
    while count>=0:
        if final_li[i][2] == count:
            lo = ([0] * (count-1)) + [1]
            mrr_list.append(lo)
        count-=1
    count = 5000

for i in range((len(col_comb)-len(final_li))):
    mrr_list.append([0])


def mean_reciprocal_rank(rs):   
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

mrr = mean_reciprocal_rank(mrr_list)
print('Mean reciprocal rank: ','\t\t\t\t', mrr)

    
# '''length check of input plus all suggestions'''
    
# length_li = []
# for i in range(len(dict_items)):
#     r = koe.encode(list(dict_items.keys())[i])
#     length_li.append([len(r), len(list(dict_items.values())[i])])
# df_length = pd.DataFrame(length_li)
# df_length = df_length.rename(columns={0: 'word length', 1 : 'suggestions'})

# mean_suggestions_number = df_length['suggestions'].sum()/len(length_li)

# mean_word_length = df_length['word length'].sum()/len(length_li)

# res1, res2 = map(list, zip(*length_li))     


# import plotly.io as pio
# pio.renderers.default = 'firefox'
# import plotly.express as px
# df = df_length
# fig = px.scatter(df, x='word length', y='suggestions')
# fig.show()









