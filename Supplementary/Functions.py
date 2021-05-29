import torch
import math

torch.set_grad_enabled(False)

from Supplementary.Modules import *

# -------------------------------------------------------------------------- #

# This function return a 2D tensor that is the rando selection of inputs for our
# stochastic gradient method, taking in count the number of mini_batches.

# We suppose here that our mini_batch_size is well chosen taking in count the fact
# that it divides input_size.
def create_random_batch(input_size, mini_batch_size):
    
    # Initialization
    N = int(input_size / mini_batch_size)
    new_batch = torch.ones(N, mini_batch_size)
    indices = torch.randperm(input_size)
    
    # Cut the tensor in N parts
    for k in range(N):
        new_batch[k] = indices[k * mini_batch_size : (k+1) * mini_batch_size]
    
    return new_batch

# -------------------------------------------------------------------------- #

# This function will train the model on nb_epochs epochs with specified
# train_input, train_classes and a mini_batch_size.
def train_model(model, train_input, train_classes, nb_epochs, mini_batch_size):
    
    for epoch in range(nb_epochs):
        # Get a N x n tensor of indices that make N "list" of n random indices
        # for the input tensor. We note n the mini_batch_size and N the number
        # of inputs divided by n. 
        random_batches = create_random_batch(train_input.size(0), mini_batch_size).long()
        
        for batch in random_batches:
            # Get the output of a sample of train_input given by the model
            output = model.forward(train_input[batch])
            # Train the model with this output tqking in count the real classes
            loss = model.backward(output, train_classes[batch])

# -------------------------------------------------------------------------- #

# This function will trained all the models contained in the lis of models in
# Models with test sets contained in Tests
def train_and_test_model(Models, train_input, train_classes, Tests, nb_epochs, mini_batch_size):
    
    # Initialization of train and test errors and standard deviation
    Train_error = []
    Test_error = []
    std_deviation = 0.0
    train_error = 0.0
    avg_nb_test_error = torch.tensor(())
    
    for k in range (0, len(Models)):
        # Renamed the actual model
        model = Models[k]
        
        # Just saying that we want Adam method for learning rate decay
        model.lr_method("Adam", 1.0e-3)
        
        train_model(model, train_input, train_classes, nb_epochs, mini_batch_size)
        
        # Set the model in evaluation mode
        model.Eval()
        
        # Compute train error
        nb_train_errors = compute_nb_errors(model, train_input, train_classes, mini_batch_size)
        train_error += nb_train_errors / 10
        
        # Compute test error
        nb_test_errors = compute_nb_errors(model, Tests[k][0], Tests[k][1], mini_batch_size)
        nb_test_errors = torch.tensor([nb_test_errors / 10]).float()
        avg_nb_test_error = torch.cat((avg_nb_test_error, nb_test_errors), 0)
        
        # Set the model in training mode
        model.Train()
    
    # Update of the train and test error
    Train_error.append(train_error / len(Models))
    Test_error.append(avg_nb_test_error.mean().tolist())

    # We just want the standard deviation after the last epoch of training
    std_deviation = avg_nb_test_error.std().tolist()
    
    return Train_error, Test_error, std_deviation

# -------------------------------------------------------------------------- #

# This function compute the number of error the model made with data_input as
# input and data_target as target taking in count if we use the cross entropy
# loss of the MSE loss.
def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        result = model.forward(data_input.narrow(0, b, mini_batch_size))
        
        if model.loss.is_MSE():
            # If the loss function is MSE
            predicted_classes = (result >= 0.5).int()
        else:
            # If the loss function is CrossEntropy
            _, predicted_classes = torch.max(result, 1)
        
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
                
    return nb_data_errors

# -------------------------------------------------------------------------- #

# This function gives train and test sets for the giving problem of the project
# that is to classifies points in [0,1]^2 if there are in a circle with a
# radius equal to R = 1 / sqrt(2pi)
def create_problem(nb_samples):
    
    # Remark: the function .uniform return a uniform distribution on [0,1) instead of [0,1],
    # but in our case it's not a problem since it is only a train and a test set on a circle
    # that do not touch the border of the set [0,1]^2.
    train_input = torch.empty(nb_samples, 2).uniform_(0, 1)
    test_input = torch.empty(nb_samples, 2).uniform_(0, 1)
    
    # Radius of our circle
    R = 1 / math.sqrt(2 * math.pi)
    
    # In this order: substract 0.5 (centered in (0,0)), take the square,
    # sum it and substract R^2 (distance of the point from the center of the 
    # circle). Now, we get the sign to know if it's outside or inside the 
    # circle and we set to zero where it's outside (substracting 1) and set to
    # 1 when it's inside (dividing by 2). The rest is for practical 
    # manipulation of the tensor for later in the algorithm. 
    train_classes = train_input.sub(0.5).pow(2).sum(1).sub(R**2).sign().sub(1).div(-2).long().resize_((nb_samples,1))
    test_classes = test_input.sub(0.5).pow(2).sum(1).sub(R**2).sign().sub(1).div(-2).long().resize_((nb_samples,1))
    
    return train_input, train_classes, test_input, test_classes

# -------------------------------------------------------------------------- #

# This function just gives a set of n test_input and test_classes, this will
# allows us to do an average test error on the trained model, to be sure that 
# the error is significant.
def get_tests(n):
    M = []
    for k in range (n):
        L = []
        _, _, test_input, test_classes =  create_problem(1000)
        L.append(test_input)
        L.append(test_classes)
        M.append(L)
    return M

# -------------------------------------------------------------------------- #

# Initialization of many different architectures using different activation
# functions or different loss. The princiapl structure will be with three 
# hidden layer of size 25 each.
def get_Models(Loss, nb_rounds):
    
    # Initialization of our models lists
    Model_ReLU_List = []
    Model_Tanh_List = []
    Model_Sigmoid_List = []
    Model_Leaky_ReLU_List = []
    Model_ELU_List = []
    Model_Tanh_Dropout_MSE = []
    Model_Tanh_Dropout_CE = []
    
    # Just change the size of the final output of or model in case we use Cross
    # Entropy loss (size = 2) or MSE loss (size = 1)
    final_out = 2
    if (Loss.is_MSE()):
        final_out = 1
    
    # Loop adding at each list a new model, there will nb_rounds number of them.
    for k in range (0, nb_rounds):
        Model_ReLU_List.append(Sequential([Linear(2,25), ReLU(), Linear(25,25), ReLU(), Linear(25,25), ReLU(), Linear(25,final_out), ReLU()], Loss))
        Model_Tanh_List.append(Sequential([Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25,25), Tanh(), Linear(25,final_out), Tanh()], Loss))
        Model_Sigmoid_List.append(Sequential([Linear(2,25), Sigmoid(), Linear(25,25), Sigmoid(), Linear(25,25), Sigmoid(), Linear(25,final_out), Sigmoid()], Loss))
        Model_Leaky_ReLU_List.append(Sequential([Linear(2,25), Leaky_ReLU(), Linear(25,25), Leaky_ReLU(), Linear(25,25), Leaky_ReLU(), Linear(25,final_out), Leaky_ReLU()], Loss))
        Model_ELU_List.append(Sequential([Linear(2,25), ELU(), Linear(25,25), ELU(), Linear(25,25), ELU(), Linear(25,final_out), ELU()], Loss))
        Model_Tanh_Dropout_MSE .append(Sequential([Linear(2,25), Tanh(), Dropout(), Linear(25,25), Tanh(), Dropout(), Linear(25,25), Tanh(), Dropout(), Linear(25,1), Tanh()], LossMSE()))
        Model_Tanh_Dropout_CE.append(Sequential([Linear(2,25), Tanh(), Dropout(), Linear(25,25), Tanh(), Dropout(), Linear(25,25), Tanh(), Dropout(), Linear(25,2), Tanh()], CrossEntropyLoss()))
        
    return Model_ReLU_List, Model_Tanh_List, Model_Sigmoid_List, Model_Leaky_ReLU_List, Model_ELU_List, Model_Tanh_Dropout_MSE, Model_Tanh_Dropout_CE

# -------------------------------------------------------------------------- #

# This function is essentialy for the plots something that is close to a 
# density probability function by the fact that each line has sum equla to one,
# and every elements are strictly positive thanks to the exponential functions.
def soft_max(data_output):
    output = data_output.to(dtype=torch.float)
    L = output.exp()
    
    # To avoid numerical error in the computation
    maxx = L.max()
    
    L = torch.div((L * maxx), (output.exp().sum(1).resize(L.size(0),1) * maxx))
    return L