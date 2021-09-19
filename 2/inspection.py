#%%

import sys 
import csv
import os
import numpy as np
import math
# %%
if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]

# %%
def count_labels(labels):

    d = {}
    for l in labels:
        if l not in d:
            d[l] = 1
        else: 
            d[l] += 1
    return d 

def entropy(labels):
    # import pdb; pdb.set_trace()
    size = len(labels)
    d = count_labels(labels)
    h = 0 
    for key in d:
        h+=(d[key]/size)*math.log2(d[key]/size)
    h*=-1
    return h

def error_rate(labels):
    size = len(labels)
    label_count = count_labels(labels)
    max_label = max(label_count.values())
    # https://www.kite.com/python/answers/how-to-find-the-max-value-in-a-dictionary-in-python
    error_count = size - max_label
    rate = error_count/size
    return rate


traindat = []
with open(input) as tsv:
    csv.reader(input, delimiter='\t')
    for row in tsv:
        traindat.append(row.split())
    tsv.close()

train = traindat[1:]
label = [i[-1] for i in train]


h = entropy(label)
e = error_rate(label)
#%% 
# write output
with open(output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['entropy: {}'.format(h)])
    writer.writerow(['error: {}'.format(e)])
