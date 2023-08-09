import numpy as np


# Bernoulli Naive Bayes
class BernoulliNB:
    def __init__(self):
        self.classes = None # classes of the target variable
        self.class_priors = None # prior probabilities of the classes
        self.feature_probs = None # probabilities of the features

    def fit(self, X, y):
        self.classes = np.unique(y) 
        num_classes = len(self.classes)
        num_features = X.shape[1]

        self.class_priors = np.zeros(num_classes)
        self.feature_probs = np.zeros((num_classes, num_features))

        for i, c in enumerate(self.classes): # i is the index of the class, c is the class
            X_c = X[y == c] # X_c is the subset of X where the target variable is equal to the class c
            num_instances = len(X_c)

            self.class_priors[i] = num_instances / len(X)

            for feature in range(num_features):
                num_positive = np.sum(X_c[:, feature] == 1) # number of instances where the feature is equal to 1
                self.feature_probs[i, feature] = (num_positive + 1) / (num_instances + 2) # Laplace smoothing



    def predict(self, X):
        predictions = []

        for x in X:
            class_scores = []

            for i, c in enumerate(self.classes): 
                class_score = np.log(self.class_priors[i])

                for feature, value in enumerate(x):
                    if value == 1:
                        class_score += np.log(self.feature_probs[i, feature]) # log of the product of the probabilities of the features
                    else:
                        class_score += np.log(1 - self.feature_probs[i, feature]) 

                class_scores.append(class_score) 

            predicted_class = self.classes[np.argmax(class_scores)]
            predictions.append(predicted_class)

        return predictions
