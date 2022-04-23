#%%
import numpy as np
import sys
import csv
# sys.path.append(".")
#%%
from datetime import datetime
startTime = datetime.now()

#%%
# implements a sentiment polarity analyzer using 
# binary logistic regression
if __name__ == '__main__':
    formatted_train_input = sys.argv[1] # path to the formatted training input .tsv file
    formatted_validation_input = sys.argv[2] # path to the formatted validation input .tsv file
    formatted_test_input = sys.argv[3] # path to the formatted test input .tsv file
    dict_input = sys.argv[4] # path to the dictionary input .txt file
    train_out = sys.argv[5] # path to output .labels file to which the prediction on the training data should be written
    test_out = sys.argv[6] # path to output .labels file to which the prediction on the test data should be written
    metrics_out = sys.argv[7] # path of the output .txt file to which metrics such as train and test error should be written 
    num_epoch = int(sys.argv[8]) # integer specifying the number of times SGD loops through all of the training data
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

        loss = []

        # Initialize weights
        # Folds in bias to the first index
        weights = np.array([0]*(X.shape[1]))

        for _ in range(epoch):
            N = X.shape[0]
            l = 0
            for i in range(N):
                # y prediction from current weight
                y_pred = self.sigmoid(np.dot(weights, X[i]))

                # update weights
                weights = self.gradient_update_sgd(weights, lr, X[i], y[i], y_pred, N)
                
                # loss per sample
                l += (1/N)*self.loss_function(weights, X[i], y[i])
            loss.append(l)
            
        # Assign weights, loss
        self.weights = weights 
        self.loss = np.array(loss)

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
def set_dictionary(dict_input):
    dict = {}
    with open(dict_input) as f:
        for line in f:
            line = line.split()
            dict[line[0]] = int(line[1])
        f.close()
    return dict 

def open_formatted_file(input_path):
    data = np.genfromtxt(fname=input_path, delimiter='\t', dtype=None, encoding='utf-8')
    X = data[:,1:]
    bias_feature = np.ones((X.shape[0],1), dtype='int32')
    X = np.append(bias_feature, X, axis=1)
    y = data[:,0]
    return X, y

def output_file(output_path, labels):
    np.savetxt(output_path, labels, delimiter='\n', fmt='%i')

#%%
# Train
train_X, train_y = open_formatted_file(formatted_train_input)
valid_X, valid_y = open_formatted_file(formatted_validation_input)
test_X, test_y = open_formatted_file(formatted_test_input)

lr = LogisticRegression()
lr.train(train_X, train_y, epoch=num_epoch, lr=0.01)
#%%
# Prediction
train_y_pred = lr.predict(train_X)
train_error = lr.error_rate(train_y, train_y_pred)

test_y_pred = lr.predict(test_X)
test_error = lr.error_rate(test_y, test_y_pred)

output_file(train_out, train_y_pred)
output_file(test_out, test_y_pred)

#%%
# Metric
with open(metrics_out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['error(train): {}'.format("{:.6f}".format(train_error))])
    writer.writerow(['error(test): {}'.format("{:.6f}".format(test_error))])

#%%
print(datetime.now() - startTime)
#%%
# Plot
import matplotlib.pyplot as plt

# %%
