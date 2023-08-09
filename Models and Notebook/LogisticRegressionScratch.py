import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000): # Hyperparameters that must be adjusted. If high, learner will take large steps leading to minimim loss function overshooting. If low, learner will take small steps and take too long to get the minimum loss. The best is too start high and then slowly decreasing so that it is a function of the iteration k
        self.learning_rate = learning_rate # If lr low, learning will take small steps and take too long to get the minimum loss.
        self.num_iterations = num_iterations # high iteration number will lead to overfitting
        self.weights = None # weights are the coefficients of the model
        self.bias = None # bias is the intercept of the model


    # Sigmoid Function
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))



    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1]) # X.shape[1] = n_features = f | # y = X * w + b => [n_samples, 1] = [n_samples, n_features] * [n_features, 1] + [n_samples, 1] ==> (m x 1) = ( m x f ) * ( f x 1 ) + ( m x 1 )
        self.bias = 0

        # Gradient descent (how are the parameters of the model, the weights w and bias b, learned?)
        for _ in range(self.num_iterations):
            # Calculate the y_pred (predicted values)
            y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias) # y = X . w + b

            # Compute the gradients
            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y ))
            db = (1 / X.shape[0]) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw # iteratively updating weight (w) with learning rate times the gradients dw
            self.bias -= self.learning_rate * db # iteratively updating bias (b) with learning rate times the gradients db


    # Calculate the predicted probabilities
    def predict(self, X):
        
        y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)  #dot product notation from linear algebra, z = X.w + b
 
        # Convert probabilities to binary predictions (0 or 1)
        predictions = np.where(y_pred >= 0.5, 1, 0) # Decision boundary 
        return predictions