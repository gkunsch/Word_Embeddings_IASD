def reduce_matrix(X_orig, dim, eigv):
    import numpy as np
    """
    Reduces the dimension of a (m × n)   matrix `X_orig` to
                          to a (m × dim) matrix `X_reduced`
    It uses only the first 100000 rows of `X_orig` to do the mapping.
    Matrix types are all `np.float32` in order to avoid unncessary copies.
    """
    if eigv is None:
        mapping_size = 100000
        X = X_orig[:mapping_size]
        X = X - X.mean(axis=0, dtype=np.float32)
        C = np.divide(np.matmul(X.T, X), X.shape[0] - 1, dtype=np.float32)
        _, U = np.linalg.eig(C)
        eigv = U[:, :dim]

    X_reduced = np.matmul(X_orig, eigv)

    return (X_reduced, eigv)


def load_data_from_path_en_fr(path, ft_fr, ft_en):
    '''
    Get the embeddings for the parallel corpus data for translation from english to french

    Input: 
        - path: relative path of the file containing the data
        - fr_fr: the embedding for French words loaded from whatever methods (here Fasttex)
        - ft_en: the embedding for English words loaded from whatever methods (here Fasttex)
    
    Output : 
        - words: 2D array of the parralel corpus composed of English words on 1st column and French words on 2nd column
        - embeddings_en
        - embeddings_fr
    '''
    
    import numpy as np
    import io 
    
    file = open(path, 'r')
    nb_of_translations = 0
    for line in file:
        nb_of_translations += 1
    
    dim_fr_embed = ft_fr.get_dimension()
    embeddings_fr = np.zeros((nb_of_translations,dim_fr_embed))
    
    dim_en_embed = ft_en.get_dimension()
    embeddings_en= np.zeros((nb_of_translations,dim_en_embed))
    
    words = np.empty((nb_of_translations,2), dtype=np.object_)

    with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
                en_word, fr_word = line.rstrip().split(' ', 1)
                words[i,:] = [en_word,fr_word]

                embedding_eng_word = ft_en.get_word_vector(en_word)
                embeddings_en[i,:] = embedding_eng_word
                
                embedding_fr_word = ft_fr.get_word_vector(fr_word)
                embeddings_fr[i,:] = embedding_fr_word
    
    return words, embeddings_en, embeddings_fr

def load_data_from_path_fr_en(path, ft_fr, ft_en):
    '''
    Get the embeddings for the parallel corpus data for translation from english to french

    Input: 
        - path: relative path of the file containing the data
        - fr_fr: the embedding for French words loaded from whatever methods (here Fasttex)
        - ft_en: the embedding for English words loaded from whatever methods (here Fasttex)
    
    Output : 
        - words: 2D array of the parralel corpus composed of French words on 1st column and English words on 2nd column
        - embeddings_en
        - embeddings_fr
    '''
    
    import numpy as np
    import io 
    
    file = open(path, 'r')
    nb_of_translations = 0
    for line in file:
        nb_of_translations += 1
    
    dim_fr_embed = ft_fr.get_dimension()
    embeddings_fr = np.zeros((nb_of_translations,dim_fr_embed))
    
    dim_en_embed = ft_en.get_dimension()
    embeddings_en= np.zeros((nb_of_translations,dim_en_embed))
    
    words = np.empty((nb_of_translations,2), dtype=np.object_)

    with io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
                en_word, fr_word = line.rstrip().split(' ', 1)
                words[i,:] = [fr_word,en_word]

                embedding_eng_word = ft_en.get_word_vector(en_word)
                embeddings_en[i,:] = embedding_eng_word
                
                embedding_fr_word = ft_fr.get_word_vector(fr_word)
                embeddings_fr[i,:] = embedding_fr_word
    
    return words, embeddings_fr, embeddings_en

def index_to_drop(matrix_embeddings):
    '''
    Get the index for which elements of matrix_embeddings are not adequate 

    Input: 
        - matrix_embeddings: 2D array of embeddings for words (one row = one word)
    
    Output : 
        - index: array of index where the words were not rightly processed by the embedding 
    '''
    import numpy as np 
    row_sums = matrix_embeddings.sum(axis=1)
    return np.where(row_sums == 0)[0]

def normalize_vector(vector): 
    import numpy as np
    return vector/np.linalg.norm(vector)

def normalize_matrix_by_row(matrix): 
    import numpy as np 
    matrix_normalized = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        matrix_normalized[i,:] = normalize_vector(matrix[i,:])
    return matrix_normalized