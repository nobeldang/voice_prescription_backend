#!/usr/bin/env python
# coding: utf-8

# In[1]:


def predict(text):
    import json
    import nltk
    import numpy as np
    from keras.utils import to_categorical
    from nltk.tokenize import sent_tokenize, word_tokenize
    from keras.preprocessing import sequence
    from keras.models import Sequential
    from keras.layers import SimpleRNN, Dense, Activation, LSTM, Embedding, Dropout
    from keras.activations import relu, tanh, sigmoid
    from keras.preprocessing import sequence
    from nltk.stem import WordNetLemmatizer 
    import pickle
    dbfile = open('model', 'rb')      
    model = pickle.load(dbfile)
    dbfile.close()
    dbfile = open('word_to_index', 'rb')      
    word_to_index = pickle.load(dbfile)
    dbfile.close()
    dbfile = open('index_to_word', 'rb')      
    index_to_word = pickle.load(dbfile)
    dbfile.close()
    dbfile = open('vocab_dict', 'rb')      
    vocab = pickle.load(dbfile)
    dbfile.close()
    
    lemmtizer=WordNetLemmatizer()
    test=sent_tokenize(text)
    x_test_words=[]
    stpwrds=set(("/","."," ","'s","'re","'ll","'r","s","re","ll","ve","'ve"))
    for i in test:
        i=i.lower()
        word_as_temp_tokens=word_tokenize(i)
        word_tokens=[lemmtizer.lemmatize(w.lower()) for w in word_as_temp_tokens if w not in stpwrds]
        for i in range(len(word_tokens)):
            if word_tokens[i] not in vocab:
                word_tokens[i]="?"
        word_tokens_final=[w for w in word_tokens if w in vocab]
        x_test_words.append(word_tokens)

    label_dict={"name":0,
           "prescription":1,
           "symptom":2,
           "advice":3}
    
    inv_label_dict={0:"name",
           1:"prescription",
           2:"symptom",
           3:"advice"}
    
    testing=[]
    for i in x_test_words:
        index_sent=[word_to_index.get(j) for j in i]
        testing.append(index_sent)

    xtest=sequence.pad_sequences(testing,20)
    
    predictions=np.argmax(model.predict(xtest),axis=1)

    ret_dic={"name":"","prescription":"","symptom":"","advice":""}
    
    for i in range(len(predictions)):
            ret_dic[inv_label_dict[predictions[i]]]= ret_dic[inv_label_dict[predictions[i]]]+test[i]+"\n"
    
    
    ret_json = json.dumps(ret_dic)
    
    return ret_json
