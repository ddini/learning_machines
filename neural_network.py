import numpy as np

class ANN(object):
    """
        Representation of (fully connected) MLP Artificial Neural Network model.
        
        An instance of ANN requires a network architecture specification.
        Once created, an ANN object can perform forward propagation given
        an input (represented as a Numpy ndarray), and backward propagation
        given an input and output ndarray, or given a batch of examples.
    """

    def __init__(self, layer_shapes, eta=0.5, alpha=0.1):
        """
            'layer_shapes' refers to the number of (fully connected)
            nodes in each layer, beginning with the input layer.

            'weights' is a list of Numpy ndarray.

            'eta' refers to learning rate.
            'alpha' refers to momentum parameter
        """
        
        #Capture network parameters
        self.eta = eta
        self.alpha = alpha
        self.layer_shapes = layer_shapes

        #Create and initialize weights
        #-----------------------------
        
        #'self.weights' is a list of ndarray matrices, with one matrix
        #per layer.
        #The matrix for layer L is of shape:
        # (L-1, L), where L-1 refers to the number of units in the
        #preceding layer. With L-1 referring to the input number of
        #dimensions, if L is the first layer following inputs.

        self.weights = []

        for i in range(len(self.layer_shapes)-1):
            
            #Create initialized weight matrix,
            #of shape (L-1, L)
            new_arr = np.random.rand(self.layer_shapes[i], self.layer_shapes[i+1])
            self.weights.append(new_arr)
        #-----------------------------
    
    def forward(self, input_vec, labels=None):
        """
            Execute forward propagation. 
            
            input: Numpy ndarray containing input values.
                Input shape is (N x p), where p is number of dimensions,
                and N is number of examples.
            
            Returns: an ndarray in the shape:
                (N x o) where o is the number of nodes in the output
                layer.
            
            If 'labels' is not None, returns a tuple:
                -ndarray in the shape of output layer
                -instantaneous error signal, in the shape
                 of output layer.
        """

        last_input = input_vec
        
        layer_output = None
        for i in range(len(self.weights)):
            
            #Calculate local field
            local_field = np.dot(last_input, self.weights[i])

            #Calculate output vector
            #-----------------------
            z = np.exp(-1*local_field)
            layer_output = np.divide(z, 1+z)
            #-----------------------

            last_input = layer_output
        
        return layer_output
    
    def backprop(self, input, label):
        pass

        #Execute forward propagation, using 'label' in
        #order to obtain instantaneous error signal.
        network_out, error = self.forward(input, labels=label)

        #Use errors to recursively calculate gradient
        #values for each weight in network.
        #For layer L to 1
            
            #Calculate dWji values

            #Store delta values
        
        #Accumulate useful statistics about learning process.

    
    def backprop_batch(self, inputs, outputs):
        """
            Calculate error signal from batch, prior to
            computing weight updates 
        """

        pass

def activation(x, act_type="sigmoid"):
    """
        Activation function for node.
        Default value is sigmoid.
    """

    if act_type == "sigmoid":
        return sigmoid_func(x)
    else:
        return None

def sigmoid_func(x):
    """
    """

    if x>=0:
        z = exp(-x)
    else:
        z = exp(x)
    
    return z / (1+z)


