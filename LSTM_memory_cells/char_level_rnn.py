from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dropout, Dense, Activation
from keras.optimizers import Adam
import keras.backend as K

import numpy as np
import sys
import pickle


class CharLevelRNN():
    X_LEN = 20  # use X_LEN chars to predict the next one
    def __init__(self, weights_file=None):     
        if weights_file:
            print('Load model from:', weights_file)           
            self.model = load_model(weights_file)
            return   
        print('To build a new model, must run \"gen_trainset_from_text\" or ' + 
            '\"load_trainset\".')        

    def gen_trainset_from_text(self, text_file):
        text = open(text_file, encoding='utf-8').read().lower()        
        chars = sorted(list(set(text)))
        self.char_index = {c: i for i, c in enumerate(chars)}
        self.index_char = {i: c for i, c in enumerate(chars)}
        self.n_char = len(self.char_index)

        X = []
        Y = []       
        for i in range(0, len(text) - self.X_LEN, 3):
            X.append(text[i: i + self.X_LEN])
            Y.append(text[i + self.X_LEN])

        m = len(X)
        x = np.zeros((m, self.X_LEN, self.n_char), dtype=np.bool)
        y = np.zeros((m, self.n_char), dtype=np.bool)
        for i, sentence in enumerate(X):
            for t, char in enumerate(sentence):
                x[i, t, self.char_index[char]] = 1
            y[i, self.char_index[Y[i]]] = 1
        self.x = x
        self.y = y
        if not hasattr(self, 'model'):
            self.model = self.get_new_model()

    def load_trainset(self, pkl_file):
        params = pickle.load(open(pkl_file, 'rb'))
        for key, value in params.items():
            setattr(self, key, value)
        self.n_char = len(self.char_index)
        if not hasattr(self, 'model'):
            self.model = self.get_new_model()

    def dump_trainset(self, pkl_file):
        params = {'char_index': self.char_index, 
            'index_char': self.index_char, 'X_LEN': self.X_LEN,
            'x': self.x, 'y': self.y}
        pickle.dump(params, open(pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def train(self, batch_size=128, epochs=1):        
        self.model.fit(self.x, self.y, batch_size=batch_size, epochs=epochs)

    def generate_text(self, starter, length = 300, temperature=1.0):
        # assert: '@' is not in self.char_index
        sentence = starter.lower().rjust(self.X_LEN, '@')
        generated = starter          
        for _ in range(length):
            x_pred = np.zeros((1, self.X_LEN, self.n_char))
            for t, char in enumerate(sentence):
                if char != '@':
                    x_pred[0, t, self.char_index[char]] = 1.
            
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self._sample(preds, temperature = temperature)
            next_char = self.index_char[next_index]
            
            generated += next_char
            sentence = sentence[1:] + next_char
        return generated

    def get_new_model(self):
        print('Build a new model.')    
        input = Input((self.X_LEN, self.n_char))
        X, _, _ = LSTM(128, return_sequences=True, return_state=True)(input)
        X = Dropout(0.5)(X)
        X, _, _ = LSTM(128, return_state=True)(X)
        X = Dropout(0.5)(X)
        X = Dense(self.n_char)(X)
        X = Activation('softmax')(X)
        model = Model(inputs=[input], outputs=[X])
        
        model.compile(optimizer=Adam(), 
            loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _sample(self, preds, temperature):
        exp_preds = np.exp(np.log(preds.astype(np.float64)) / temperature)       
        softmax = exp_preds / np.sum(exp_preds)        
        return np.random.choice(range(self.n_char), p = softmax)        


if __name__ == '__main__':
    rnn = CharLevelRNN('models/patent_1M_40.h5')
    #rnn.gen_trainset_from_text('data/patent_1M.txt')
    #rnn.dump_trainset('data/patent_1M.pkl')
    rnn.load_trainset('data/patent_1M.pkl')
    #rnn.train(epochs=1)
    #rnn.model.save('models/patent_1M_40.h5')
    print(rnn.generate_text('<?xml version=\"1.0\" '))
