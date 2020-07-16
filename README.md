# learning_machines
Learning machine approaches and variants.

### Multilayer Perceptron

aMLP = MultilayerPerceptron((1,15,1), eta=0.001)

X = np.arange(0,50, 0.1)
X = np.reshape(X, (X.shape[0],1))
Y = np.power(X,3)
Y_train = Y/max(Y)
Y_train = np.reshape(Y_train, (Y_train.shape[0],1))
MLP.fit(X, Y_train, num_iterations=500)
max(Y)*MLP.predict(X)
