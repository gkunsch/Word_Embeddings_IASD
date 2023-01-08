def get_nn_scores(embedding_model_source, word, words_data, tgt_emb, W, K=5, epsilon = 1e-7):
    '''
    Computes and gives back the K nearest neighbors, as well as their scores, for the translation of specific word from english to french

    Inputs: 
        - embedding_model: bin, the embedding model 
        - word: string, particular word to translate
        - words_data: 2D array of corpus of parallel data (1st column english, 2nd column French)
        - tgt_emb: 2D array, matrix of embeddings for the french words in words_data
        - W: 2D array, optimal matrix obtained from an optimisation method 
        - K: int, number of nearest neighbors to display
        - epsilon: float, small parameter for numerical stability

    Outputs: 
        - The display of K nearest neighbors with their scores
    '''

    import numpy as np 
    
    print("Nearest neighbors of \"%s\":" % word)
    
    word_emb = embedding_model_source.get_word_vector(word)
    vector_translation = np.dot(word_emb, W)

    scores = (tgt_emb / (np.linalg.norm(tgt_emb, 2, 1)[:, None]+epsilon)).dot(vector_translation / (np.linalg.norm(vector_translation)+epsilon)) #cosine similarity
    k_best = scores.argsort()[-K:][::-1]
    
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], words_data[idx,1])) 

def get_nn(embedding_model_source, word, words_data, tgt_emb, W, K=5, epsilon = 1e-7):
    '''
    Computes and gives back the K nearest neighbors, as well as their scores, for the translation of specific word from english to french

    Inputs: 
        - embedding_model: bin, the embedding model 
        - word: string, particular word to translate
        - words_data: 2D array of corpus of parallel data (1st column english, 2nd column French)
        - tgt_emb: 2D array, matrix of embeddings for the french words in words_data
        - W: 2D array, optimal matrix obtained from an optimisation method 
        - K: int, number of nearest neighbors to display
        - epsilon: float, small parameter for numerical stability

    Outputs: 
        - The display of K nearest neighbors without their scores
    '''
    import numpy as np 

    word_emb = embedding_model_source.get_word_vector(word)
    vector_translation = np.dot(word_emb, W)

    scores = (tgt_emb / (np.linalg.norm(tgt_emb, 2, 1)[:, None]+epsilon)).dot(vector_translation / (np.linalg.norm(vector_translation)+epsilon)) #cosine similarity
    k_best = scores.argsort()[-K:][::-1]
    
    likely_K_words = []
    for i, idx in enumerate(k_best): 
        likely_K_words.append(words_data[idx,1])
    return likely_K_words


def test_translation(embedding_model_source, words_test, tgt_emb, W, K=5, limit= 3000): 
    '''
    Computes for each word in the test set the translation it haves and assign a 1 if the translation if right, 0 otherwise

    Inputs: 
        - words_test: 2D array of corpus of parallel data (1st column english, 2nd column French)
        - tgt_emb: 2D array, matrix of embeddings for the french words in words_data
        - K: int, number of nearest neighbors to find the translation into
        - limit: int, number in case it is too long to process on the whole dataset (around 2800 samples in dataset)
    Outputs: 
        - The display of K nearest neighbors without their scores
    '''

    array_of_translation = [] #1 if good translation, 0 otherwise
    i = 0
    while (i < len(words_test) and i<limit): #to get rid of the last empty line in .txt file
        word_to_test = words_test[i,0] #english word
        if  words_test[i,1] in get_nn(embedding_model_source = embedding_model_source, word = word_to_test, words_data = words_test, tgt_emb = tgt_emb, W = W, K=K, epsilon = 1e-7): 
            array_of_translation.append(1)
        else: 
            array_of_translation.append(0)
        i = i+1
    
    return array_of_translation