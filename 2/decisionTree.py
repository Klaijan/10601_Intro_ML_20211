#%%
import sys
import csv 
import math

# %%
if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_output = sys.argv[4]
    test_output = sys.argv[5]
    metrics_output = sys.argv[6]

#%%

# Functions 
# label_set will define after open the input train file as global variable

# count the labels as dict, key with 0 counts also available in the dictionary
def count_labels(labels):
    d = {}
    for label in label_set:
        d[label] = 0
    for l in labels:
        d[l] += 1
    return d 

# calculate the entropy h(y) from the input dataset
# the dataset input will varies in size/length as it is prefiltered according to the attributed before calling the function
def entropy(dataset):
    labels = [i[-1] for i in dataset]
    size = len(dataset)
    d = count_labels(labels)
    h = 0 
    for key in d:
        # try except to catch log(0) case
        try:
            h+=(d[key]/size)*math.log2(d[key]/size)
        except:
            pass
    h*=-1
    return h

# calculate conditional entropy h(y|x)
def conditional_entropy(dataset, attr_idx):
    cond_h = {}
    attr_val = [i[attr_idx] for i in dataset]
    valset = list(set(attr_val))
    attr_size = len(attr_val)
    for attr_label in valset:
        slices_data = [i for i in dataset if i[attr_idx] == attr_label]
        size = len(slices_data)
        d = {}
        for label in label_set:
            d[label] = 0
        for row in slices_data:
            d[row[-1]] += 1
        h = 0
        for key in d:
            if d[key] != 0:
                h+=(d[key]/size)*math.log2(d[key]/size)
        h*=-1
        cond_h[attr_label] = {"prob": size/attr_size, "entropy": h, "lc":d}
    return cond_h

# calculate mutual information
# takes in hy entropy as numerical and hy_x conditional entropy as list of dictionary
def mutual_information(hy, hy_x):
    i_xy = [hy-x[i]['hx'] for i, x in enumerate(hy_x)]

    # handling the tie mutual information case
    # split at the first attribute
    max_info_attr_idx = 0
    for idx in range(len(i_xy)):
        if i_xy[idx] > i_xy[max_info_attr_idx]:
            max_info_attr_idx = idx
    return i_xy, max_info_attr_idx, hy_x

# pretty print the tree output
# takes in node and the label from dataset input
def pprint(current_node, label):
    lc = count_labels(label)
    if current_node.depth == 0:
        print("[", end="") 
        print("/".join("{} {}".format(str(value), str(key)) for key, value in lc.items()), end="") 
        print("]")
    else:
        lc = "/".join("{} {}".format(str(value), str(key)) for key, value in lc.items())
        print("{}{} = {}: [{}]".format("|"*current_node.depth, current_node.attr, current_node.condition, lc))

# perform majority vote
def majority_vote(dataset):
    # takes in label_count as dict and return the max value (key)
    # of tie, choose the last lexicographical order
    labels = [i[-1] for i in dataset]
    label_count = count_labels(labels)

    if len(set(label_count.values())) == 1:
        return list(label_count)[-1]
    else:
        return max(label_count, key=label_count.get)

#%%
# Node/tree related functions

# create tree from root node
def split(current_node, dataset, attr_name):
    label = [i[-1] for i in dataset]

    # print the tree as it learns
    pprint(current_node, label)
 
    # terminal state to use majority vote 
    # (and also handle tie majority vote case)
    # reach max depth, depth = 0, no attr left
    if current_node.depth == max_depth or max_depth == 0:
        current_node.label = majority_vote(dataset)
        return

    hy = entropy(dataset)
    hy_x = []
    for attr_idx in range(0,len(dataset[0])-1):
        hx_i = conditional_entropy(dataset, attr_idx)
        hx = 0
        for i in hx_i:
            hx += hx_i[i]['prob']*hx_i[i]['entropy']
        hy_x.append({attr_idx: {"hx": hx, "hx_i": hx_i}})
    
    # information gain
    i_xy, split_idx, hy_x = mutual_information(hy, hy_x)

    # when information gain = 0
    if i_xy[split_idx] == 0:
        current_node.label = majority_vote(dataset)
        return 

    # split at split_idx if information gain is not 0
    else:
        split_attr_val = list(hy_x[split_idx][split_idx]['hx_i'].keys())
        # attr val = value of the attribution, will use as decision
        # e.g. 'y','n' 'A','notA'

        # if no split = only one value
        if len(split_attr_val) == 1:
            current_node.label = majority_vote(dataset)
            return
        else:
            left_dataset = [row for row in dataset if row[split_idx] == split_attr_val[0]]
            right_dataset = [row for row in dataset if row[split_idx] == split_attr_val[1]]

            next_depth = current_node.depth + 1
            split_on_attr = attr_name[split_idx]

            left_child = Node(next_depth, split_on_attr, split_idx, split_attr_val[0])
            right_child = Node(next_depth, split_on_attr, split_idx, split_attr_val[1]) 

            current_node.left = left_child
            current_node.right = right_child

            split(left_child, left_dataset, attr_name)
            split(right_child, right_dataset, attr_name)
#%%
# Class node
class Node:
    def __init__(self, depth, attr, attr_idx, condition):
        self.depth = depth
        self.attr = attr
        self.attr_idx = attr_idx
        self.condition = condition
        self.label = None

        self.left = None 
        self.right = None
        

# %%
# I/O
# input = """C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw2\\Code\\mushroom_train.tsv"""
train = []
with open(train_input) as tsv:
    csv.reader(train_input, delimiter='\t')
    for row in tsv:
        train.append(row.split())
    tsv.close()

attr_name = train[0]
train = train[1:]
train_label = [i[-1] for i in train]
label_set = sorted(list(set(train_label)))
label_count = count_labels(train_label)

# input = """C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw2\\Code\\mushroom_test.tsv"""

test = []
with open(test_input) as tsv:
    csv.reader(test_input, delimiter='\t')
    for row in tsv:
        test.append(row.split())
    tsv.close()

test = test[1:]
test_label = [i[-1] for i in test]

# start create tree
root = Node(0, None, None, None)
split(root, train, attr_name)

#%%
# prediction
def predict(node, row):
    # if done at level = 0
    if node.label is not None:
        return node.label
    elif row[node.left.attr_idx] == node.left.condition:
        if node.left.label is not None:
            return node.left.label
        else:
            return predict(node.left, row)
    elif row[node.right.attr_idx] == node.right.condition:
        if node.right.label is not None:
            return node.right.label
        else:
            return predict(node.right, row)

def prediction(tree, dataset):
    answers = []
    for row in dataset:
        answers.append(predict(tree, row))
    return answers

#%%   
# predict train
train_pred = prediction(root, train)
# predict test
test_pred = prediction(root, test)
# %%
def error_rate(truth, pred):
    error = 0
    for t,p in zip(truth,pred):
        if t != p:
            error+=1
    return error/len(truth)

train_error = error_rate(train_label, train_pred)
test_error = error_rate(test_label, test_pred)

# %%
# write output
# train out
with open(train_output, 'w', newline='') as f:
    writer = csv.writer(f)
    for t in train_pred:
        writer.writerow([t])
# test out
with open(test_output, 'w', newline='') as f:
    writer = csv.writer(f)
    for t in test_pred:
        writer.writerow([t])
# metrics out
with open(metrics_output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['error(train): {}'.format("{:.6f}".format(train_error))])
    writer.writerow(['error(test): {}'.format("{:.6f}".format(test_error))])
