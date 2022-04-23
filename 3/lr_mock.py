#%%
import numpy as np
import csv

#%%
# implements a sentiment polarity analyzer using 
# binary logistic regression

#%%
# dictionary 
dictionary = {}
dict_path = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\handout\\dict.txt'
with open(dict_path) as f:
    for line in f:
        line = line.split()
        dictionary[line[0]] = int(line[1])
    f.close()
#%%
# takes in formatted tsv

inputpath = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\Code\\model1_formatted_train_l.tsv'
# inputpath = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\Code\\model1_formatted_train.tsv'
data = np.genfromtxt(fname=inputpath, delimiter='\t', dtype=None, encoding='utf-8')
X = data[:,1:]
bias_feature = np.ones((X.shape[0],1), dtype='int32')
X = np.append(bias_feature, X, axis=1)
# add bias feature to X?
y = data[:,0]

#%%
class LogisticRegression:
    def sigmoid(self, z):
        # takes in weights as vector (M+1x1), x as vector (M+1x1)
        return 1/(1+np.e**(-z))

    def loss_function(self, weights, x_i, y):
        y_true = y
        y_pred = self.sigmoid(np.dot(weights, x_i))
        return -((y_true*np.log(y_pred)) + (1-y_true)*np.log(1-y_pred))

    def gradient_update_sgd(self, weights, lr, x_i, y_true, y_pred, N):
        weights = weights + (1/N)*lr*(y_true-y_pred)*x_i
        return weights

    def train(self, X, y, epoch, lr):
        # takes X training input as nparray size (NxM+1) where N = data points and M = # features and +1 is bias parameter
        # takes y label as vector size (Nx1) shape (N, )?

        all_weights = []

        # Initialize weights
        # Folds in bias to the first index
        weights = np.array([0]*(X.shape[1]))

        for _ in range(epoch):
            print("epoch",_)
            N = X.shape[0]
            w = []
            for i in range(N):
                # y prediction from current weight
                y_pred = self.sigmoid(np.dot(weights, X[i]))

                # update weights
                weights = self.gradient_update_sgd(weights, lr, X[i], y[i], y_pred, N)
                w.append(weights)
            all_weights.append(weights)
                # loss per sample
                # l += (1/N)*self.loss_function(weights, X[i], y[i])
            # loss.append(l)
            
        print("Finished!")
        # return weights, loss
        self.weights = weights 
        self.all_weights = np.array(all_weights)
        # self.loss = np.array(loss)

    def predict(self, X):
        # return 1 or 0
        z = self.sigmoid(np.dot(X, self.weights))
        return np.where(z >= 0.5, 1, 0)
    
    def error_rate(self, truth, pred):
        error = 0
        for t,p in zip(truth,pred):
            if t != p:
                error+=1
        return error/len(truth)


#%%
lr = 0.001
lr_train = LogisticRegression()
lr_train.train(X, y, epoch=5000, lr=lr)
print(lr)
all_weights = lr_train.all_weights

#%%
train_loss = lr_train.loss
#%%
# losses = []
#%%
# losses.append(train_loss)
#%%
def dl(losses):
    l1 = []
    for loss in losses:
        l2 = []
        for i in range(len(loss)-1):
            l2.append(loss[i+1]-loss[i])
        l1.append(l2)
    return l1
losses_dl = dl(losses)
#%%
# Prediction
# train
train_y_pred = lr_train.predict(X)
error = lr_train.error_rate(y, train_y_pred)
accuracy = 1-error
# print(accuracy)
print(error)
#%%
inputpath = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\Code\\model1_formatted_valid_l.tsv'
# inputpath = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\Code\\model1_formatted_train.tsv'
data = np.genfromtxt(fname=inputpath, delimiter='\t', dtype=None, encoding='utf-8')
test_X = data[:,1:]
bias_feature = np.ones((test_X.shape[0],1), dtype='int32')
test_X = np.append(bias_feature, test_X, axis=1)
# add bias feature to X?
test_y = data[:,0]

#%%

# test 

# inputpath = 'C:\\Users\\Acer\\Study Materials\\2021_1 Fall\\10601 Introduction to Machine Learning\\HW\\hw4\\hw4\\Code\\model2_formatted_test.tsv'
# data = np.genfromtxt(fname=inputpath, delimiter='\t', dtype=None, encoding='utf-8')
# X = data[:,1:]
# bias_feature = np.ones((X.shape[0],1), dtype='int32')
# X = np.append(bias_feature, X, axis=1)
# # add bias feature to X?
# y = data[:,0]

# y_pred = lr.predict(X)
# # Error
# accuracy = lr.error_rate(y, y_pred)
# error = 1-accuracy
# print(accuracy)
# print(error)
# # %%
# def output_file(output_path, labels):
#     with open(output_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         for l in labels:
#             writer.writerow(l)
# #%%
# # train out
# output_file(train_out, y_pred)

# train_error = lr.error_rate(y, y_pred)
# test_error = lr.error_rate(y, y_pred)

# with open(output_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['error(train): {}'.format("{:.6f}".format(train_error))])
#     writer.writerow(['error(test): {}'.format("{:.6f}".format(test_error))])

# # %%
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.plot(losses[0], label='train')
plt.plot(losses[1], label='valid')
# plt.plot(losses[2], label='0.1')
plt.xlabel("epochs")
plt.ylabel("average negative log likelihood")
plt.legend(loc='best')
plt.title("Model 1")
#%%
plt.figure(figsize=(12,5))
plt.plot(losses[0], label='0.001')
plt.plot(losses[1], label='0.01')
plt.plot(losses[2], label='0.1')
plt.xlabel("epochs")
plt.ylabel("average negative log likelihood")
plt.legend(loc='best')
plt.title("Model 1")
# %%
plt.figure(figsize=(12,5))
plt.plot(losses_dl[0], label='0.001')
plt.plot(losses_dl[1], label='0.01')
plt.plot(losses_dl[2], label='0.1')
plt.xlabel("epochs")
plt.ylabel("average negative log likelihood changes")
plt.legend(loc='best')
plt.title("Model 1")
# %%
plt.plot(losses_dl[2], label='0.001', color='green')
plt.xlabel("epochs")
plt.ylabel("average negative log likelihood changes")
plt.legend(loc='best')

# %%
plt.plot(losses_dl[1], label='0.01', color='orange')
plt.xlabel("epochs")
plt.ylabel("average negative log likelihood changes")
plt.legend(loc='best')

#%%
plt.plot(losses_dl[0], label='0.1')
plt.xlabel("epochs")
plt.ylabel("average negative log likelihood changes")
plt.legend(loc='best')

# %%
def sigmoid(z):
    # takes in weights as vector (M+1x1), x as vector (M+1x1)
    return 1/(1+np.e**(-z))

def loss_function(weights, x_i, y, N):
    y_true = y
    y_pred = sigmoid(np.dot(weights, x_i))
    return -(1/N)*((y_true*np.log(y_pred)) + (1-y_true)*np.log(1-y_pred))

loss_train_001 = []
loss_val_001 = []
#%%
for n in range(5000):
    print("epoch",n)
    l1 = 0
    l2 = 0
    for i in range(X.shape[0]):
        l1+=loss_function(all_weights[n], X[i], y[i], X.shape[0])
    loss_train_001.append(l1)
    # print(l1)
    # for i in range(test_X.shape[0]):
    #     l2+=loss_function(all_weights[n], test_X[i], test_y[i], test_X.shape[0])
    # print(l2)
    # loss_val_001.append(l2)

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(losses_dl[0], label='0.1')
plt.plot(losses_dl[1], label='0.01')
plt.plot(losses_dl[2], label='0.001')
plt.xlabel("epochs")
plt.ylabel("average negative log likelihood")
plt.legend(loc='best')
plt.title("Model 1")
# %%
# losses = []
losses.append(loss_train_001)
# %%
0.1
0.01
0.001
