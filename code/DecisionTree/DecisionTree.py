"""
Implementation d'un arbre de decision from scratch.
Utilise l'impurete de Gini pour les splits.
"""
import numpy as np
import random

class TreeNode:
    """Noeud de l'arbre de decision."""

    def __init__(self, val=None, feature=None, threshold=None, left=None, right=None):
        self.val = val
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = threshold
        self.data = None

    def calculate_gini(self, data):
        """Calcule l'impurete de Gini pour un ensemble de donnees."""
        true_val = data[data[:, data.shape[1]-1] == 1]
        false_val = data[data[:, data.shape[1]-1] == 0]
        return 1 - (len(true_val)/len(data))**2 - (len(false_val)/len(data))**2

    def fit(self, data, depth, max_depth, min_samples_split, last_gini, min_gini_change, n_features):
        """Construit l'arbre recursivement en cherchant le meilleur split."""
        count_true = len(data[data[:, data.shape[1]-1] == 1])
        count_false = len(data[data[:, data.shape[1]-1] == 0])

        # Conditions d'arret
        if depth >= max_depth or count_true == 0 or count_false == 0 or data.shape[0] < min_samples_split:
            if count_true >= count_false:
                self.val = 1
            else:
                self.val = 0
            return self

        lowest_gini = float("inf")
        lowest_gini_feature = -1
        lowest_gini_threshold = -1

        # Selection aleatoire de features (pour Random Forest)
        flag_features = False
        loop_features = data.shape[1] - 1
        indices = set(range(data.shape[1]-1))
        idx = 0

        if n_features < data.shape[1] - 1:
            flag_features = True
            loop_features = n_features

        for k in range(0, loop_features):
            if flag_features:
                random_idx = random.choice(tuple(indices))
                indices.remove(random_idx)
                idx = random_idx
            else:
                idx = k

            unique_values = np.unique(data[:, idx])

            # Cherche le meilleur seuil pour cette feature
            for j in range(unique_values.shape[0]-1):
                filter_value = (unique_values[j]+unique_values[j+1])/2

                left = data[data[:, idx] <= filter_value]
                right = data[data[:, idx] > filter_value]
                assert len(left) + len(right) == len(data)

                total_gini_impurity = (len(left)/len(data))*self.calculate_gini(left) + (len(right)/len(data))*self.calculate_gini(right)

                if total_gini_impurity < lowest_gini:
                    lowest_gini = total_gini_impurity
                    lowest_gini_feature = idx
                    lowest_gini_threshold = filter_value

        left = data[data[:, lowest_gini_feature] < lowest_gini_threshold]
        right = data[data[:, lowest_gini_feature] >= lowest_gini_threshold]

        # Regularisation sur le changement minimum de Gini
        if abs(lowest_gini-last_gini) < min_gini_change:
            if count_true >= count_false:
                self.val = 1
            else:
                self.val = 0
            return self

        self.data = data
        self.feature = lowest_gini_feature
        self.threshold = lowest_gini_threshold

        self.left = TreeNode().fit(data=left, depth=depth+1, max_depth=max_depth,
                                   min_samples_split=min_samples_split, last_gini=lowest_gini,
                                   min_gini_change=min_gini_change, n_features=n_features)
        self.right = TreeNode().fit(data=right, depth=depth+1, max_depth=max_depth,
                                    min_samples_split=min_samples_split, last_gini=lowest_gini,
                                    min_gini_change=min_gini_change, n_features=n_features)

        return self


class DecisionTree:
    """Arbre de decision avec support du bagging."""

    def __init__(self, tree=None, max_depth=float("inf"), min_samples_split=0, min_gini_change=-1, bagging=False):
        self.tree = tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gini_change = min_gini_change
        self.bagging = bagging

    def perform_bagging(self, data):
        """Echantillonnage bootstrap des donnees."""
        new_data = np.empty(data.shape, dtype=object)

        for i in range(new_data.shape[0]):
            random_row = data[np.random.randint(data.shape[0])]
            new_data[i] = random_row

        assert new_data.shape == data.shape
        return new_data

    def build_tree(self, data, n_features=float("inf")):
        """Construit l'arbre a partir des donnees."""
        if self.tree is not None:
            raise Exception("Error in building decision tree. Tree is not None.")

        else:
            if self.bagging:
                final_data = self.perform_bagging(data)
            else:
                final_data = data

            head = TreeNode()
            head = head.fit(
                data=final_data,
                depth=0,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gini_change=self.min_gini_change,
                last_gini=1,
                n_features = n_features
            )

            self.tree = head

    def predict(self, data):
        """Predit les labels pour un ensemble de donnees."""
        assert data.shape[0] != 0
        assert data.shape[0] != 1

        results = np.full((data.shape[0], 1), -1)

        for idx, sample in enumerate(data):
            sample_d = sample

            if sample.shape[0] != 1:
                sample_d = np.expand_dims(sample_d, axis=0)

            result = self.predict_sample(sample_d)
            results[idx] = result

        return results

    def predict_sample(self, sample):
        """Predit le label pour un seul echantillon."""
        assert sample.shape[0] == 1

        def predict_sample_recursive(node, depth=0):
            assert depth < self.max_depth+1

            if node.val is not None:
                return node.val

            if sample[0][node.feature] <= node.threshold:
                return predict_sample_recursive(node.left, depth + 1)
            else:
                return predict_sample_recursive(node.right, depth + 1)

        return predict_sample_recursive(self.tree)

    def calculate_accuracy(self, data):
        """Calcule la precision sur un jeu de donnees."""
        labels = data[:, data.shape[1]-1:]
        samples = data[:, :data.shape[1]-1]

        results = self.predict(samples)

        return np.mean(results == labels)


    def print_tree(self):
        """Affiche l'arbre dans la console."""
        def print_recursive(node, depth=0):
            if node is None:
                return

            indent = "  " * depth

            if node.val is not None:
                print(f"{indent}Leaf: {node.val}")

            else:
                print(f"{indent}Feature {node.feature} < {node.threshold}")
                print(f"{indent}Left:")
                print_recursive(node.left, depth + 1)
                print(f"{indent}Right:")
                print_recursive(node.right, depth + 1)

        print_recursive(self.tree)
