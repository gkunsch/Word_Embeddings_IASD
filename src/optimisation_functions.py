def gradient_descent(X_train,Y_train, X_test, Y_test, nb_of_iterations, step_size, early_stopping = False):
    '''
    Perform Whole Gradient Descent and return optimal parameters 
    as well as evolution of loss functions for training and test sets for plotting

    Inputs:
        - X_train: 2D array of source embeddings for training
        - Y_train: 2D array of target embeddings for training
        - X_test: 2D array of source embeddings for test
        - Y_test: 2D array of target embeddings for tets
        - nb_of_iterations: int, number of iterations (epoch for Whole GD)
        - step_size: float, fixed learning rate for parameters update 
        - early_stopping: boolean, stop the process if the test loss function goes up too much

    Outputs:
        - W: 2D array, linear trasnlation from the target to the source space
        - iteration: array, composed of the int corresponding to the iteration
        - loss_function_train_array: array, value of loss function on the train set at each iteration
        - loss_function_test_array: array, value of loss function on the test set at each iteration
    '''

    import numpy as np 

    #We initialize W (parameters of the model)
    W = np.random.rand(X_train.shape[1],Y_train.shape[1])
    N_TRAIN = X_train.shape[0]
    N_TEST = X_test.shape[0]
    loss_function_train_array= []
    loss_function_test_array =[]
    iteration = []

    for i in range(nb_of_iterations): 

        iteration.append(i)
        
        loss_function_train = (1/N_TRAIN)*np.linalg.norm(np.dot(X_train,W)-Y_train)**2 #RMSE loss
        loss_function_train_array.append(loss_function_train)
        
        loss_function_test = (1/N_TEST)*np.linalg.norm(np.dot(X_test,W)-Y_test)**2 #RMSE loss
        loss_function_test_array.append(loss_function_test)
        
        grad = (2/N_TRAIN)*X_train.T.dot(np.dot(X_train,W)-Y_train)

        W = W - step_size*grad

        if (early_stopping and len(loss_function_test_array) > 3 and loss_function_test_array[-1] > loss_function_test_array[-2] and loss_function_test_array[-1] > loss_function_test_array[-3]):
            nb_of_iterations = i+1
            break

    return W, iteration, loss_function_train_array, loss_function_test_array

def batch_gradient_descent(X_train,Y_train, X_test, Y_test, nb_of_iterations, step_size, batch_percentage, early_stopping = False):
    '''
    Perform Batch Gradient Descent and return optimal parameters 
    as well as evolution of loss functions for training and test sets for plotting

    Inputs:
        - X_train: 2D array of source embeddings for training
        - Y_train: 2D array of target embeddings for training
        - X_test: 2D array of source embeddings for test
        - Y_test: 2D array of target embeddings for tets
        - nb_of_iterations: int, number of iterations (epoch for Whole GD)
        - step_size: float, fixed learning rate for parameters update 
        - batch_percentage: percentage of data in the training set that should be included in the gradient calculation 
        - early_stopping: boolean, stop the process if the test loss function goes up too much

    Outputs:
        - W: 2D array, linear trasnlation from the target to the source space
        - iteration: array, composed of the int corresponding to the iteration
        - loss_function_train_array: array, value of loss function on the train set at each iteration
        - loss_function_test_array: array, value of loss function on the test set at each iteration
    '''

    import numpy as np 

    #We initialize W (parameters of the model)
    W = np.random.rand(X_train.shape[1],Y_train.shape[1])
    N_TRAIN = X_train.shape[0]
    batch_size = round(N_TRAIN*batch_percentage)
    N_TEST = X_test.shape[0]
    loss_function_train_array= []
    loss_function_test_array =[]
    iteration = []

    for i in range(nb_of_iterations): 

        iteration.append(i)
        
        loss_function_train = (1/N_TRAIN)*np.linalg.norm(np.dot(X_train,W)-Y_train)**2 #RMSE loss
        loss_function_train_array.append(loss_function_train)
        
        loss_function_test = (1/N_TEST)*np.linalg.norm(np.dot(X_test,W)-Y_test)**2 #RMSE loss
        loss_function_test_array.append(loss_function_test)
        
        idx_to_select = np.random.choice(N_TRAIN,batch_size,replace=True)
        X_train_batch = X_train[idx_to_select,:]
        Y_train_batch = Y_train[idx_to_select,:]
        
        grad = (2/batch_size)*X_train_batch.T.dot(np.dot(X_train_batch,W)-Y_train_batch)

        W = W - step_size*grad

        if (early_stopping and len(loss_function_test_array) > 3 and loss_function_test_array[-1] > loss_function_test_array[-2] and loss_function_test_array[-1] > loss_function_test_array[-3]):
            nb_of_iterations = i+1
            break

    return W, iteration, loss_function_train_array, loss_function_test_array



def Procustes(W): 
    import numpy as np 
    U, S, V = np.linalg.svd(W)
    W_orthogonal = U@V
    return W_orthogonal 

def check_orthogonal(W):
    import numpy as np
    n = W.shape[0]
    if (W@W.T).all() == np.eye(n).all(): 
        return True 
    else: 
        return False

def objective_cosine_function(W,X,Y): 
    import numpy as np
    sum=0
    for i in range(X.shape[0]):
        sum = sum + np.dot(np.dot(X[i,:],W),Y[i,:])
    return sum 

def vector_multiplication_transpose(X_vect,Y_vect):
    import numpy as np 
    M = np.zeros((len(X_vect), len(Y_vect)))
    for i in range(len(X_vect)):
        for j in range(len(Y_vect)):
            M[i,j] = X_vect[i]*Y_vect[j]
    return M 

def grad_compute_cosine(X,Y):
    import numpy as np 
    W_grad = np.zeros((X.shape[1], Y.shape[1]))
    for i in range(X.shape[0]):
        W_grad = W_grad + vector_multiplication_transpose(X[i,:],Y[i,:])
    return W_grad


def batch_gradient_descent_full_procustes(X_train,Y_train, X_test, Y_test, nb_of_iterations, step_size, batch_percentage, early_stopping = False):
    '''
    Perform GD on the whole dataset and return the plot of the training loss with respect to the number of iterations
    '''
    import numpy as np 
    #We initialize W (parameters of the model)
    W = np.random.rand(X_train.shape[1],Y_train.shape[1])
    
    N_TRAIN = X_train.shape[0]
    batch_size = round(N_TRAIN*batch_percentage)
    N_TEST = X_test.shape[0]
    loss_function_train_array= []
    loss_function_test_array =[]
    iteration = []

    for i in range(nb_of_iterations): 
        iteration.append(i)
        
        loss_function_train = (1/N_TRAIN)*objective_cosine_function(W,X_train,Y_train) #cosine loss
        loss_function_train_array.append(loss_function_train)
        
        loss_function_test = (1/N_TEST)*objective_cosine_function(W,X_test,Y_test) #cosine loss
        loss_function_test_array.append(loss_function_test)
        
        idx_to_select = np.random.choice(N_TRAIN,batch_size,replace=True)
        X_train_batch = X_train[idx_to_select,:]
        Y_train_batch = Y_train[idx_to_select,:]
        
        grad = (1/batch_size)*grad_compute_cosine(X_train_batch,Y_train_batch)

        W = W + step_size*grad #max problem

        W = Procustes(W) #made orthogonal        

        #if not check_orthogonal(W):
        #    break
        
        if (early_stopping and len(loss_function_test_array) > 3 and loss_function_test_array[-1] < loss_function_test_array[-2] and loss_function_test_array[-1] < loss_function_test_array[-3]):
            nb_of_iterations = i+1
            break

    return W, iteration, loss_function_train_array, loss_function_test_array

def gradient_descent_projected(X_train,Y_train, X_test, Y_test, nb_of_iterations, step_size, early_stopping = False):
    '''
    Perform Whole GD on the dataset and return optimal parameters 
    as well as evolution of loss functions for training and test sets for plotting
    '''

    import numpy as np 
    #We initialize W (parameters of the model)
    W = np.random.rand(X_train.shape[1],Y_train.shape[1])
    N_TRAIN = X_train.shape[0]
    N_TEST = X_test.shape[0]
    loss_function_train_array= []
    loss_function_test_array =[]
    iteration = []

    for i in range(nb_of_iterations): 

        iteration.append(i)
        
        loss_function_train = (1/N_TRAIN)*np.linalg.norm(np.dot(X_train,W)-Y_train)**2 #RMSE loss
        loss_function_train_array.append(loss_function_train)
        
        loss_function_test = (1/N_TEST)*np.linalg.norm(np.dot(X_test,W)-Y_test)**2 #RMSE loss
        loss_function_test_array.append(loss_function_test)
        
        grad = (2/N_TRAIN)*X_train.T.dot(np.dot(X_train,W)-Y_train)

        W = W - step_size*grad
        W = Procustes(W) #orthogonal function

        if (early_stopping and len(loss_function_test_array) > 3 and loss_function_test_array[-1] > loss_function_test_array[-2] and loss_function_test_array[-1] > loss_function_test_array[-3]):
            nb_of_iterations = i+1
            break

    return W, iteration, loss_function_train_array, loss_function_test_array

def batch_gradient_descent_projected(X_train,Y_train, X_test, Y_test, nb_of_iterations, step_size, batch_percentage, early_stopping = False):
    '''
    Perform Batch GD on the dataset and return optimal parameters 
    as well as evolution of loss functions for training and test sets for plotting
    '''

    import numpy as np 
    #We initialize W (parameters of the model)
    W = np.random.rand(X_train.shape[1],Y_train.shape[1])
    N_TRAIN = X_train.shape[0]
    batch_size = round(N_TRAIN*batch_percentage)
    N_TEST = X_test.shape[0]
    loss_function_train_array= []
    loss_function_test_array =[]
    iteration = []

    for i in range(nb_of_iterations): 

        iteration.append(i)
        
        loss_function_train = (1/N_TRAIN)*np.linalg.norm(np.dot(X_train,W)-Y_train)**2 #RMSE loss
        loss_function_train_array.append(loss_function_train)
        
        loss_function_test = (1/N_TEST)*np.linalg.norm(np.dot(X_test,W)-Y_test)**2 #RMSE loss
        loss_function_test_array.append(loss_function_test)
        
        idx_to_select = np.random.choice(N_TRAIN,batch_size,replace=True)
        X_train_batch = X_train[idx_to_select,:]
        Y_train_batch = Y_train[idx_to_select,:]
        
        grad = (2/batch_size)*X_train_batch.T.dot(np.dot(X_train_batch,W)-Y_train_batch)

        W = W - step_size*grad
        W = Procustes(W)

        if (early_stopping and len(loss_function_test_array) > 3 and loss_function_test_array[-1] > loss_function_test_array[-2] and loss_function_test_array[-1] > loss_function_test_array[-3]):
            nb_of_iterations = i+1
            break

    return W, iteration, loss_function_train_array, loss_function_test_array

