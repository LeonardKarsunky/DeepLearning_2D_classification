import torch
import math

from Supplementary.Modules import *
from Supplementary.Functions import *

# Just use for the plots at the end
import matplotlib.pyplot as plt
from pylab import imshow, colorbar

torch.set_grad_enabled(False)

# -------------------------------------------------------------------------- #
# INITIALIZATION
# -------------------------------------------------------------------------- #

# Initialization of our principal parameters
nb_epochs = 100
mini_batch_size = 10

# This gives the nb of models that we will train and test to have conclusive
# results.
# ! IMPORTANT ! 
# If you want to run the file more faster, it is advisable to 
# reduce the number of rounds equal to 2.
nb_rounds = 10

# Initialization of training set
train_input, train_classes, _, _ = create_problem(1000)

# initialization of our test sets
Tests = get_tests(nb_rounds)

# -------------------------------------------------------------------------- #
# MSE
# -------------------------------------------------------------------------- #

# initialization of all the models with MSE loss
Model_ReLU_MSE, Model_Tanh_MSE, Model_Sigmoid_MSE, Model_Leaky_ReLU_MSE, Model_ELU_MSE, _, _ = \
                                get_Models(LossMSE(), nb_rounds)

# -------------------------------------------------------------------------- #

# Activation function as ReLU, with MSE loss
Train_error_ReLU_MSE, Test_error_ReLU_MSE, std_deviation_ReLU_MSE = \
    train_and_test_model(Model_ReLU_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with ReLU and LossMSE 
print("Average train_error on", nb_rounds,"Sequential models with ReLU and LossMSE is",Train_error_ReLU_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with ReLU and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with ReLU and LossMSE is",Test_error_ReLU_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_ReLU_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as Tanh, with MSE loss
Train_error_Tanh_MSE, Test_error_Tanh_MSE, std_deviation_Tanh_MSE = \
    train_and_test_model(Model_Tanh_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Tanh and LossMSE
print("Average train_error on", nb_rounds,"Sequential models with Tanh and LossMSE is",Train_error_Tanh_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Tanh and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with Tanh and LossMSE is",Test_error_Tanh_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Tanh_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as a Sigmoid, with MSE loss
Train_error_Sigmoid_MSE, Test_error_Sigmoid_MSE, std_deviation_Sigmoid_MSE = \
    train_and_test_model(Model_Sigmoid_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Sigmoid and LossMSE 
print("Average train_error on", nb_rounds,"Sequential models with Sigmoid and LossMSE is",Train_error_Sigmoid_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Sigmoid and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with Sigmoid and LossMSE is",Test_error_Sigmoid_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Sigmoid_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as a Leaky ReLU, with MSE loss
Train_error_Leaky_ReLU_MSE, Test_error_Leaky_ReLU_MSE, std_deviation_Leaky_ReLU_MSE = \
    train_and_test_model(Model_Leaky_ReLU_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Leaky_ReLU and LossMSE 
print("Average train_error on", nb_rounds,"Sequential models with Leaky_ReLU and LossMSE is",Train_error_Leaky_ReLU_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Leaky_ReLU and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with Leaky_ReLU and LossMSE is",Test_error_Leaky_ReLU_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Leaky_ReLU_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as ELU, with MSE loss
Train_error_ELU_MSE, Test_error_ELU_MSE, std_deviation_ELU_MSE = \
    train_and_test_model(Model_ELU_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with ELU and LossMSE 
print("Average train_error on", nb_rounds,"Sequential models with ELU and LossMSE is",Train_error_ELU_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with ELU and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with ELU and LossMSE is",Test_error_ELU_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_ELU_MSE, "%")

# -------------------------------------------------------------------------- #
# CROSS ENTROPY AND DROPOUT
# -------------------------------------------------------------------------- #

# initialization of all the models with Cross Entropy (CE) loss and dropout models
_, Model_Tanh_CE, _, _, _, Model_Tanh_Dropout_MSE, Model_Tanh_Dropout_CE = \
                                get_Models(CrossEntropyLoss(), nb_rounds)

# -------------------------------------------------------------------------- #

# Activation function as Tanh, with Cross Entropy (CE) loss
Train_error_Tanh_CE, Test_error_Tanh_CE, std_deviation_Tanh_CE = \
    train_and_test_model(Model_Tanh_CE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Tanh and Cross-Entropy Loss
print("Average train_error on", nb_rounds,"Sequential models with Tanh and Cross-Entropy Loss is",Train_error_Tanh_CE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Tanh and Cross-Entropy Loss
print("Average test_error on", nb_rounds,"Sequential models with Tanh and Cross-Entropy Loss is",Test_error_Tanh_CE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Tanh_CE, "%")

# -------------------------------------------------------------------------- #

# Activation function as Tanh, with MSE loss and dropout
Train_error_Tanh_Dropout_MSE, Test_error_Tanh_Dropout_MSE, std_deviation_Tanh_Dropout_MSE = \
    train_and_test_model(Model_Tanh_Dropout_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Tanh and Cross-Entropy Loss
print("Average train_error on", nb_rounds,"Tanh with Dropout and MSE Loss is",Train_error_Tanh_Dropout_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Tanh and Cross-Entropy Loss
print("Average test_error on", nb_rounds,"Tanh with Dropout and MSE Loss is",Test_error_Tanh_Dropout_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Tanh_Dropout_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as Tanh, with Cross Entropy (CE) loss and dropout
Train_error_Tanh_Dropout_CE, Test_error_Tanh_Dropout_CE, std_deviation_Tanh_Dropout_CE = \
    train_and_test_model(Model_Tanh_Dropout_CE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Tanh and Cross-Entropy Loss
print("Average train_error on", nb_rounds,"Tanh with Dropout and Cross-Entropy Loss is",Train_error_Tanh_Dropout_CE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Tanh and Cross-Entropy Loss
print("Average test_error on", nb_rounds,"Tanh with Dropout and Cross-Entropy Loss is",Test_error_Tanh_Dropout_CE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Tanh_Dropout_CE, "%")

# -------------------------------------------------------------------------- #
# FINAL PLOTS
# -------------------------------------------------------------------------- #

# Initialization of our problem
train_input, train_classes, _, _ = create_problem(1000)
nb_epochs = 100
mini_batch_size = 1

# Initialization of the model with Cross Entropy loss
model = Sequential([Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25,2), Tanh()], CrossEntropyLoss())
model.lr_method("Adam", 1.0e-3)
train_model(model, train_input, train_classes, nb_epochs, mini_batch_size)

# Initialization of the grid for the plots
x = torch.linspace(0.0, 1.0, 1000)
y = torch.linspace(0.0, 1.0, 1000)
n = x.size(0)
grid_x, grid_y = torch.meshgrid(x,y)
Z = torch.empty(grid_x.size())

# Get the "probability" that each point is in the circle, given by the output 
# of the model going through the soft_max function.
for k in range(grid_x.size(0)):
    for l in range(0, n, mini_batch_size):        
        xx = grid_x[k][l:l+mini_batch_size].t().resize_((mini_batch_size,1))
        yy = grid_y[k][l:l+mini_batch_size].t().resize_((mini_batch_size,1))
        e = torch.cat([xx, yy], 1)
        
        if model.loss.is_MSE():
            # MSE Loss
            Z[k][l:l+mini_batch_size] = model.forward(e).resize(mini_batch_size)
        else:
            # Cross Entropy loss
            output = model.forward(e)
            output = soft_max(output)
            
            # We take the second colum because we want the "probability" that 
            # the point is in the circle. It is given in the second column.
            second_column = torch.tensor([1])
            Z[k][l:l+mini_batch_size] = output.index_select(1,second_column).resize(mini_batch_size)



# First figure
plt.figure(1)
figure, axes = plt.subplots()
color_map = plt.cm.get_cmap('RdYlBu')
reversed_color_map = color_map.reversed()

# Plot of the tensor
im = imshow(Z, origin='lower', extent=[0,1,0,1], cmap=reversed_color_map)

# Plot of the true circle of radius R
axes = plt.gca()
R = 1 / math.sqrt(2 * math.pi)
Drawing_uncolored_circle = plt.Circle((0.5, 0.5 ), R, fill=False, linestyle = '--', label='Target circle')
axes.set_aspect(1)
axes.add_artist(Drawing_uncolored_circle)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of the results given by the trained model')
plt.legend()
plt.legend(handles=[Drawing_uncolored_circle], loc='upper right')

colorbar(im)
plt.savefig('Final_results.jpg')


# Second figure
plt.figure(2)
Label = train_classes.view(-1).float()
x1 = (train_input.narrow(1,0,1).view(-1) * Label);
y1 = (train_input.narrow(1,1,1).view(-1) * Label);
x0 = (train_input.narrow(1,0,1).view(-1) * (1-Label));
y0 = (train_input.narrow(1,1,1).view(-1) * (1-Label));
plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')

plt.gca().set_aspect('equal', adjustable='box')
axes = plt.gca();
Lab0, = plt.plot(x1, y1, 'C3o', label='Label 0');
Lab1, = plt.plot(x0, y0, 'C0o', label='Label 1');
plt.xlabel('x')
plt.ylabel('y')

# Plot of the true circle of radius R
R = 1 / math.sqrt(2 * math.pi)
Drawing_uncolored_circle = plt.Circle((0.5, 0.5 ), R, fill=False, linestyle = '--', label='Target circle')
axes.set_aspect(1)
axes.add_artist(Drawing_uncolored_circle)

plt.title('Plot of train set and the target cicle')
plt.legend()
plt.legend(handles=[Lab1, Lab0, Drawing_uncolored_circle], loc='upper right')
axes.set_xlim([0,1]);
axes.set_ylim([0,1]);
plt.savefig('Train_set.jpg')
