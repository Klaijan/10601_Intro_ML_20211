#%%
import numpy as np
import sys
#%%
class Feature:
    def __init__(self):
        pass
#%%
# Dictionary 
    def set_dictionary(self, dict_input):
        dict = {}
        with open(dict_input) as f:
            for line in f:
                line = line.split()
                k = line[0]
                v = int(line[1])
                # if feature_flag == 2: v = [float(l) for l in line[1:]]
                dict[k] = v
            f.close()
        self.dict = dict
    def set_feature_dictionary(self, feature_dictionary_input):
        fdict = {}
        with open(feature_dictionary_input) as f:
            for line in f:
                line = line.split()
                k = line[0]
                v = [float(l) for l in line[1:]]
                fdict[k] = v
            f.close()
        self.fdict = fdict
    def set_feature_flag(self, feature_flag):
        self.feature_flag = feature_flag

#%%
# Feature Engineering

    # Tokenization
    def tokenization(self, text):
        # Takes text and tokenize them into list of words
        if len(text) == 0:
            return []
        else:
            return text.split()

    #%%
    # Model 1
    # Bag of word
    def bow_representation(self, token, label):
        bow_vec = np.array([0]*len(self.dict))
        for t in token:
            if t in self.dict.keys():
                bow_vec[self.dict[t]] = 1
        # Return nparray of shape M+1 where M is the dictionary size
        # First array column is label
        return np.append(label, bow_vec)
    # %%
    # Model 2
    # Word Embedding
    def wordemb_representation(self, token, label):
        trimmed = []
        emb_vec = []
        for t in token:
            if t in self.fdict.keys():
                # emb_vec.append((t, dict[t]))
                trimmed.append(t)
                emb_vec.append(self.fdict[t])
        emb_vec = np.array(emb_vec)
        emb = (1/emb_vec.shape[0])*np.sum(emb_vec, axis=0)
        # Round to 6 decimal places
        emb = np.round(emb, 6)
        # Return nparray of shape 301
        # First array column is label
        return np.append(label, emb)
    
    def format_output(self, data, output_path):
        # output_path = output_path.split('/')
        # output_path[-1] = 'model'+str(self.feature_flag)+'_'+output_path[-1]
        # output_path = ('/').join(output_path)

        token = [self.tokenization(d['tdata']) for d in data]

        if self.feature_flag == 1:
            features = np.array([self.bow_representation(t, l) for t,l in zip(token, data['label'])])
            np.savetxt(output_path,features,delimiter='\t', fmt='%i')
        elif self.feature_flag == 2:
            features = np.array([self.wordemb_representation(t, l) for t,l in zip(token, data['label'])])
            np.savetxt(output_path,features,delimiter='\t', fmt='%.6f')
    
    def open_file(self, input_path):
        label = []
        tdata = []
        with open(input_path) as f:
            for row in f:
                s = row.split('\t')
                label.append(int(s[0]))
                tdata.append(s[1])
            f.close()
        data = np.empty(len(label), dtype={'names':('label', 'tdata'),
                          'formats':('int', object)})
        data['label'] = label 
        data['tdata'] = tdata
        return data
#%%
#%%
if __name__ == '__main__':
    train_input = sys.argv[1] # path to the training input .tsv file
    validation_input = sys.argv[2] # path to the validation input .tsv file
    test_input = sys.argv[3] # path to the test input .tsv file
    dict_input = sys.argv[4] # path to the dictionary input .txt file
    formatted_train_out = sys.argv[5] # path to output .tsv file to which the feature extractions on the training data should be written
    formatted_validation_out = sys.argv[6] # path to output .tsv file to which the feature extractions on the validation data should be written
    formatted_test_out = sys.argv[7] # path to output .tsv file to which the feature extractions on the test data should be written
    feature_flag = int(sys.argv[8]) # integer taking value 1 or 2 that specifies whether to construct the Model 1 feature set or the Model 2 feature set
    feature_dictionary_input = sys.argv[9] # path to the word2vec feature dictionary .tsv file

    f = Feature()
    f.set_dictionary(dict_input)
    f.set_feature_dictionary(feature_dictionary_input)
    f.set_feature_flag(feature_flag)

    # Open input data file 
    train_data = f.open_file(train_input)
    val_data = f.open_file(validation_input)
    test_data = f.open_file(test_input)

    f.format_output(train_data, output_path=formatted_train_out)
    f.format_output(val_data, output_path=formatted_validation_out)
    f.format_output(test_data, output_path=formatted_test_out)