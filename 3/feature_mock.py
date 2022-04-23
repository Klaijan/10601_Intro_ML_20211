#%%
import numpy as np
import csv
from numpy.lib.function_base import diff

from numpy.lib.utils import deprecate_with_doc
#%%
# define dictionary and word2vec 
# https://stackoverflow.com/questions/14505898/how-to-make-a-dictionary-from-a-text-file-with-python

dictionary = {}
dict_path = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\handout\\dict.txt'
with open(dict_path) as f:
    for line in f:
        line = line.split()
        dictionary[line[0]] = int(line[1])
    f.close()

wordvec = {}
wordvec_path = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\handout\\word2vec.txt'
with open(wordvec_path) as f:
    for line in f:
        line = line.split()
        word = line[0]
        vec = [float(l) for l in line[1:]]
        wordvec[word] = vec
    f.close()

#%%
inputpath = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\handout\\smalldata\\train_data.tsv'
label = []
tdata = []
with open(inputpath) as f:
    for row in f:
        s = row.split('\t')
        label.append(int(s[0]))
        tdata.append(s[1])
    f.close()
#%%
data = np.empty(len(label), dtype={'names':('label', 'tdata'),
                          'formats':('int', object)})
data['label'] = label 
data['tdata'] = tdata
#%%
# inputpath = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\handout\\smalldata\\train_data.tsv'
# data = np.genfromtxt(fname=inputpath, names=['label','data'], delimiter='\t', dtype=None, encoding='utf-8')

#%%
# Feature Engineering

# Model 1
# Bag of word

def tokenization(text):
    # Takes text and tokenize them into list of words
    if len(text) == 0:
        return []
    else:
        return text.split()

def bow_representation(dict, token, label):
    bow_vec = np.array([0]*len(dict))
    for t in token:
        if t in dict.keys():
            # print(t)
            # print(dict[t])
            bow_vec[dict[t]] = 1
            # print(bow_vec[dict[t]])
            # print("=====================")
    # Return nparray of shape M+1 where M is the dictionary size
    # First array column is label
    return np.append(label, bow_vec)

# %%
# Model 2
# Word Embedding
def wordemb_representation(dict, token, label):
    trimmed = []
    emb_vec = []
    for t in token:
        if t in dict.keys():
            # emb_vec.append((t, dict[t]))
            trimmed.append(t)
            emb_vec.append(dict[t])
    emb_vec = np.array(emb_vec)
    # print(len(trimmed))
    # print(emb_vec.shape)
    emb = (1/len(trimmed))*np.sum(emb_vec, axis=0)
    # print(emb.shape)
    # Round to 6 decimal places
    emb = np.round(emb, 6)
    # Return nparray of shape 301
    # First array column is label
    return np.append(label, emb)

# def save_file(path):
#     with open(path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         for t in train_pred:
#             writer.writerow([t])
# %%
token = [tokenization(d['tdata']) for d in data]

model = 2
formattedpath = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\Code\\model{}_formatted_train.tsv'.format(model)

if model == 1:
    features = np.array([bow_representation(dictionary, t, l) for t,l in zip(token, data['label'])])
    np.savetxt(formattedpath,features,delimiter='\t', fmt='%i')
    # bow_features = np.array([bow_representation(dictionary, t, l) for t,l in zip(token, label)])
elif model == 2:
    features = np.array([wordemb_representation(wordvec, t, l) for t,l in zip(token, data['label'])])
    np.savetxt(formattedpath,features,delimiter='\t', fmt='%.6f')
    # emb_features = np.array([wordemb_representation(wordvec, t, l) for t,l in zip(token, label)])
# %%
# Format output
# formattedpath = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\Code\\model{}_formatted_train.tsv'.format(model)
# with open(formattedpath, 'w', newline='') as f:
#     writer = csv.writer(f)
#     for t in features:
#         writer.writerow(t)

#%%

#%%
import numpy as np
#%%
path = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\Code\\model2_formatted_test.tsv'
myfeat = np.genfromtxt(fname=path, delimiter='\t', dtype=None, encoding='utf-8')
myfeat_X = myfeat[:,1:]
bias_feature = np.ones((myfeat_X.shape[0],1), dtype='int32')
myfeat_X = np.append(bias_feature, myfeat_X, axis=1)
# add bias feature to X?
myfeat_y = myfeat[:,0]

#%%
path = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\handout\\smalloutput\\model2_formatted_test.tsv'
handout = np.genfromtxt(fname=path, delimiter='\t', dtype=None, encoding='utf-8')
handout_X = handout[:,1:]
bias_feature = np.ones((handout_X.shape[0],1), dtype='int32')
handout_X = np.append(bias_feature, handout_X, axis=1)
# add bias feature to X?
handout_y = handout[:,0]
# %%
print(sum(myfeat_X-handout_X))
print(sum(myfeat_y-handout_y))
# %%
x = myfeat_X-handout_X
diff_row = []
for i in range(len(x)):
    for j in range(len(x[i])):
        if x[i][j] != 0:
            print("row",i,"col",j,"vl",x[i][j])
            diff_row.append(i)
diff_row = list(set(diff_row))
# %%
