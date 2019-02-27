import numpy as np

class ANN(object):
    """
        Representation of (fully connected) MLP Artificial Neural Network model.
        
        An instance of ANN requires a network architecture specification.
        Once created, an ANN object can perform forward propagation given
        an input (represented as a Numpy ndarray), and backward propagation
        given an input and output ndarray, or given a batch of examples.
    """

    def __init__(self, layer_shapes, eta=0.5, alpha=0.1, sigmoid_scale=1.0):
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
        self.sig_scale = sigmoid_scale
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
    
    def forward(self, input_vec, labels=None, rec_acts=False):
        """
            Execute forward propagation. 
            
            input: Numpy ndarray containing input values.
                Input shape is (N x p), where p is number of dimensions,
                and N is number of examples.
            
            Returns: 
                -A dictionary containing:
                    'output':
                        an ndarray in the shape:
                        (N x o) where o is the number of nodes in the output
                        layer.
            
                    If 'labels' is not None, "errors":
                        -ndarray in the shape of output layer
                        -instantaneous error signal, in the shape
                        of output layer.
                    If 'rec_acts' is True, "activations":
                        -A list of ndarrays,
                        in which each element is the activation values
                        for the corresponding layer.
        """

        last_input = input_vec
        
        layer_output = None
        
        #*list* of ndarrays, representing
        #activation values for correspondign layer.
        activations = []

        for i in range(len(self.weights)):
            
            #Calculate local field
            local_field = np.dot(last_input, self.weights[i])

            #Calculate output vector
            #-----------------------
            z = np.exp(-1*self.sig_scale*local_field)
            
            layer_output = np.divide(z, 1+z)
            #-----------------------

            #Collect activation layers
            #-------------------------
            if rec_acts:
                activations.append(np.array(layer_output))
            #-------------------------

            last_input = layer_output

        #Compile output
        output_dict = {}

        if labels is not None:
            output_dict["error"] = np.subtract(labels, layer_output)
        
        if rec_acts:
            output_dict["activations"] = activations
        
        output_dict["output"] = layer_output

        return output_dict
    
    def backprop(self, input, label):

        #Execute forward propagation, using 'label' in
        #order to obtain instantaneous error signal.
        network_out, error = self.forward(input, labels=label, rec_acts=True)
        
        for i in reversed(range(len(self.weights))):
            pass
            #Calculate delta for this layer

            #Obtain dW values

            #Perform update
        
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

def activation_prime(x, act_type="sigmoid"):
    """
    """

    if act_type == "sigmoid":
        return sigmoid_prime(x)
    else:
        return None

def sigmoid_prime(x):
    """
    """


def sigmoid_func(x, a=1.0):
    """
    """

    if x>=0:
        z = exp(-a*x)
    else:
        z = exp(a*x)
    
    return z / (1+z)


