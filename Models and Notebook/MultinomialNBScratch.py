import numpy as np

class MultinomialNB:
    def __init__(self):
        self.classes = None # classes of the target variable
        self.class_counts = None # counts of the classes
        self.feature_counts = None # counts of the features 
        self.class_probabilities = None # probabilities of the classes
        self.feature_probabilities = None # probabilities of the features

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_counts = np.zeros(n_classes) 
        self.feature_counts = np.zeros((n_classes, n_features))
        self.class_probabilities = np.zeros(n_classes)
        self.feature_probabilities = np.zeros((n_classes, n_features))

        # Count occurrences of classes and features
        for i, c in enumerate(self.classes):
            class_instances = X[y == c]
            self.class_counts[i] = len(class_instances)
            self.feature_counts[i] = np.sum(class_instances, axis=0)

        # Calculate class probabilities
        self.class_probabilities = self.class_counts / np.sum(self.class_counts)

        # Calculate feature probabilities
        feature_sums = np.sum(self.feature_counts, axis=1, keepdims=True)
        self.feature_probabilities = (self.feature_counts + 1) / (feature_sums + n_features) #laplace smoothing

    def predict(self, X):
        y_pred = []
        for instance in X:
            log_probs = np.log(self.class_probabilities)
            for i, value in enumerate(instance):
                if value > 0:
                    log_probs += np.log(self.feature_probabilities[:, i]) * value
            y_pred.append(self.classes[np.argmax(log_probs)])
        return np.array(y_pred)
