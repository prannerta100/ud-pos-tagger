import pyconll
from collections import Counter
#Thanks Udacity NLP ND for some of the routines!

def create_dataset(UD_PATH, LANG_DIR, PREFIX):
    """
    Read all the conll files (test, dev, train) and 
    
    Args:
    UD_PATH:  the universal dependencies main folder (viz. C:/Users/DELL/Downloads/ud-treebanks-v2.6/ud-treebanks-v2.6/)
    LANG_DIR: the subfolder in UD_PATH containing all the relevant conll data (viz. 'UD_Afrikaans-AfriBooms/', 
              'UD_Persian-Seraji/', etc.
    PREFIX:   the prefix before each train/test/dev conll file, viz. 'af_afribooms-ud-', 'fa_seraji-ud-';
              the UD files have a standard format like af_afribooms-ud-train.conllu, af_afribooms-ud-dev.conllu, 
              af_afribooms-ud-test.conllu
    
    Returns:
    tag_list:     list of all tags encountered, concatenated into a single list
    word_list:    list of all words/punctuations encountered, concatenated into a single list
    X_train:      list of all training+dev sentences, where each sentence is in turn a list of the words and punctuations in it.
    Y_train:      list of all training+dev tags, corresponding to each sentence in X_train, same format as X_train
    X_test:       list of all test sentences, where each sentence is in turn a list of the words and punctuations in it.
    Y_test:       list of all test tags, corresponding to each sentence in X_test, same format as X_test
    """
    train_conll = pyconll.load_from_file(UD_PATH+LANG_DIR+PREFIX+'train.conllu')
    word_list = []
    tag_list = []
    X_train = []
    Y_train = []
    #train data
    for sentence in train_conll:
        for word in sentence:
            word_list.append(word.lemma)
            tag_list.append(word.upos)
        X_train.append([word.lemma for word in sentence])
        Y_train.append([word.upos for word in sentence])
    #dev data
    #if dev file exists, then fuse it with the training dataset. Otherwise no worries.
    #We do this because our version of HMM doesn't involve iterative training.
    try:
        dev_conll = pyconll.load_from_file(UD_PATH+LANG_DIR+PREFIX+'dev.conllu')
        for sentence in dev_conll:
            for word in sentence:
                word_list.append(word.lemma)
                tag_list.append(word.upos)
        X_train.append([word.lemma for word in sentence])
        Y_train.append([word.upos for word in sentence])
    except:
        print("Dev set not found, no worries! Train set must be good enough.")
    #test data
    test_conll = pyconll.load_from_file(UD_PATH+LANG_DIR+PREFIX+'test.conllu')
    X_test = []
    Y_test = []
    for sentence in test_conll:
        X_test.append([word.lemma for word in sentence])
        Y_test.append([word.upos for word in sentence])
    #return 
    return tag_list, word_list, X_train, Y_train, X_test, Y_test

def replace_unknown(sequence, vocab):
    """
    Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in vocab else 'nan' for w in sequence]

def simplify_decoding(X, model, vocab):
    """
    X should be a 1-D sequence of observations for the model to predict
    """
    _, state_path = model.viterbi(replace_unknown(X, vocab))
    return [state[1].name for state in state_path[1:-1]] # do not show the start/end state predictions

def pair_counts(sequences_A, sequences_B):
    """
    Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.
    
    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    counts_dict = dict()
    for i,tag in enumerate(sequences_A):
        if tag not in counts_dict:
            counts_dict[tag] = dict()
        if sequences_B[i] not in counts_dict[tag]:
            counts_dict[tag][sequences_B[i]] = 0
        counts_dict[tag][sequences_B[i]] += 1
    return counts_dict # return C(t_i, w_i)
    