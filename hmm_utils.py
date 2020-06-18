from data_prep import *
def accuracy(X, Y, model, vocab):
    """Calculate the prediction accuracy by using the model to decode each sequence
    in the input X and comparing the prediction with the true labels in Y.
    
    The X should be an array whose first dimension is the number of sentences to test,
    and each element of the array should be an iterable of the words in the sequence.
    The arrays X and Y should have the exact same shape.
    
    X = [("See", "Spot", "run"), ("Run", "Spot", "run", "fast"), ...]
    Y = [[(VERB,NN,...),...]
    """
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        try:
            most_likely_tags = simplify_decoding(observations, model, vocab)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
#        print("Yo")
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions