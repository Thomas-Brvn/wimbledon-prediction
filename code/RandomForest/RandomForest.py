"""
Implementation d'une Random Forest from scratch.
Utilise un ensemble d'arbres de decision avec bagging et selection aleatoire de features.
"""
import numpy as np
from DecisionTree import DecisionTree


class RandomForest:
    """Foret aleatoire : ensemble d'arbres de decision."""

    def __init__(self, n_features, n_estimators=100, tree_params=dict(max_depth=20, min_samples_split=10, bagging=True), bagging=True):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.tree_params = tree_params
        self.estimators = []

    def build_forest(self, data):
        """Construit la foret en entrainant n_estimators arbres."""
        for _ in range(self.n_estimators):
            new_tree = DecisionTree.DecisionTree(**self.tree_params)
            new_tree.build_tree(data, self.n_features)
            self.estimators.append(new_tree)

    def predict(self, data, print_predictions=False):
        """Predit par vote majoritaire des arbres."""
        for idx, forest in enumerate(self.estimators):
            prediction = forest.predict(data)

            if idx == 0:
                results = prediction
            else:
                results = np.append(results, prediction, axis=1)

        output_array = np.apply_along_axis(lambda col: np.bincount(col).argmax(), axis=1, arr=results)
        output_array = output_array.reshape(-1, 1)

        if print_predictions:
            print(results)

        return output_array

    def calculate_accuracy(self, data):
        """Calcule la precision sur un jeu de donnees."""
        labels = data[:, data.shape[1]-1:]
        samples = data[:, :data.shape[1]-1]

        results = self.predict(samples)

        return np.mean(results == labels)
