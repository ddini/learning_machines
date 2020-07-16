import numpy as np
import functools

class MultiLayerPerceptron:
    """
        Python and Numpy implementation of
        feedforward neural network.
    """

    def __init__(self, layers=None):
        """
            Layers is a list of integers,
            indicating size of layers, starting
            with input layer
        """

        self.layers = layers

        # Initialize weights
        self.weights = []
        for i in range(len(layers)-1):
            weights.append(np.random.normal(size=(layers[i+1], layers[i]+1)) )
        
    def reinitialize_weights(self):
        self.weights = []
        for i in range(len(layers)-1):
            weights.append(np.random.normal(size=(layers[i+1], layers[i]+1)) )
    
    def forward_prop(self, inputs, activations=False):
        output = []
        
        y = np.hstack( (np.ones( (inputs.shape[0],1)), inputs ) )
        if activations:
            output.append(y)
        
        for l_idx in range(1,len(self.layers)):
            v = np.dot(self.weights[l_idx-1], y.T)
            y = np.divide(1, 1+np.exp(-1*v))
            
            y = y.T
            y_size = y.shape
            
            if l_idx==len(self.layers)-1:
                output.append(y)
            else:
                y = np.hstack( (np.ones( (y.shape[0],1) ),y ) )
                if activations:
                    output.append(y)
        
        return output
    
    def backward_prop(self, inputs, targets, eta=0.03, as_batch=False):
        """
            One pass of backward propagation.
            'inputs' could be a single input, or a batch of input vectors
            
            if 'as_batch' is false, then computes update for each
            input vector individually
        """
        
        # forward prop with inputs
        Y = forward_prop(inputs, activations=True)

        # Compute delta values for output layer
        delta = (targets-Y[-1][0, :])*Y[-1][0, :]*(1-Y[-1][0, :])

        updates = []

        for l_idx in reversed(range(len(self.layers)-1)):
            # Second to last layer down to input layer
            # l_idx goes from len(layers)-1 down to 0

            # Compute updates for weights connecting this
            # layer to next layer
            dW = np.zeros( (self.layers[l_idx+1], self.layers[l_idx]+1) )
            for i in range(self.layers[l_idx]+1):
                # For each weight starting from a node
                # in this level, including the bias node
                #
                for j in range(self.layers[l_idx+1]):
                    # For each node in the next layer
                    # Compute update for weight i-->j
                    #print("update: %s" % str(eta*delta[0,j]*Y[l_idx][0,i]))
                    dW[j,i] = eta*delta[0,j]*Y[l_idx][0,i]


            # Compute delta quantities for this layer
            new_delta = np.zeros((1, self.layers[l_idx]) )
            for i in range(self.layers[l_idx]):
                y_prime = Y[l_idx][0,i]*(1-Y[l_idx][0,i])
                new_delta[0,i] = y_prime*functools.reduce(lambda acc,c: acc + c[0]*c[1],
                                        zip( [self.weights[l_idx][x, i+1] for x in range(self.layers[l_idx+1])],
                                                delta[0,:]
                                        ), 0)
            delta = new_delta
            #print("delta: %s" % delta)
            
            updates.append(dW)

        # Apply update
        for m_idx in range(len(updates)):
            self.weights[ len(self.weights)-1-m_idx ]+=updates[m_idx]
    
    def epoch(X,Y, num_iterations=100):
        for _ in range(num_iterations):
            for idx in range(X.shape[0]):
                backward_prop(X[idx:idx+1], Y[idx:idx+1])
    
    def fit(self, X,Y, num_epochs=1, num_iterations=100):
        for _ in range(num_epochs):
            self.epoch(X,Y, num_iterations)

    def predict(self, X):
        pass
