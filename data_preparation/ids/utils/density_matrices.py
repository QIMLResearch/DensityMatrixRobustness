from pennylane import numpy as np
import pandas as pd
import tensorflow as tf


class DensityMatrix:
    def __init__(self, df_probs, non_feature_columns=None, mean_centering=False, prob_mode=None, seed=0):
        if non_feature_columns is None:
            non_feature_columns = []
        self.mean_centering = mean_centering
        self.df = df_probs
        self.flows = {}
        self.non_feature_columns = non_feature_columns
        self.density_matrices = None
        self.prob_mode = prob_mode
        self.seed = seed

    def convert_and_normalize(self, x):
        x = np.array(x, requires_grad=False)
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    
    def get_state_probabilities(self, X):
        # if no probs are given, then construct uniform distribution of packets
        return [1/len(X) for x in X]

    def get_density_matrix(self, X, X_probs=None):
        # return 1 mixed DM using all datapoints
        X_states = [self.convert_and_normalize(x) for x in X]
        if X_probs is None:
            X_probs = self.get_state_probabilities(X)

        rho_mixed = sum(
            p * np.dot(x[np.newaxis].T, x[np.newaxis], requires_grad=False)
            for x, p in zip(X_states, X_probs)
        )
        return rho_mixed
    
    def is_mean_centered(self, matrix, tolerance=1e-6):
        column_means = np.mean(matrix, axis=0)
        return np.all(np.abs(column_means) < tolerance)

    def mean_center_matrix(self, matrix):
        column_means = np.mean(matrix, axis=0)
        mean_centered_matrix = matrix - column_means
        return mean_centered_matrix
    
    def construct_flows(self):
        columns_to_drop = [col for col in self.non_feature_columns if col in self.df.columns]
        df_flows = self.df.groupby("flow_window")
        for flow_number, df_flow in df_flows:
            probs = df_flow['dm_prob_softmax'].values
            labels = df_flow['label'].values
            X = df_flow.drop(columns_to_drop + ['dm_prob_softmax', 'label'], axis=1).values
            self.flows[flow_number] = (X, probs, labels)

    def construct_ml_dataset(self):
        # if self.mean_centering:
        #     X = np.array([self.mean_center_matrix(self.get_density_matrix(flow[0], flow[1])) for flow in self.flows.values()])
        # else:
        if self.prob_mode == 'uniform':
            X = np.array([self.get_density_matrix(flow[0], None) for flow in self.flows.values()])
        elif self.prob_mode == 'rand':
            X = np.array([self.get_density_matrix(flow[0], self.generate_random_list(len(flow[0]), self.seed)) for flow in self.flows.values()])
        else:
            X = np.array([self.get_density_matrix(flow[0], flow[1]) for flow in self.flows.values()])
        y = np.array([flow[2] for flow in self.flows.values()], dtype=object)
        return X, y
    
    def get_test_train_split(self, X, y, split=None):

        def collapse_labels(labels):
            return np.array([np.bincount(label).argmax() for label in labels])

        y = collapse_labels(y)
        idx = np.where(y == 1)[0][0]

        X_train = X[:idx]
        X_test = X[idx:]

        y_train = y[:idx]
        y_test = y[idx:]

        return X_train, X_test, y_train, y_test

    def main(self):
        self.construct_flows()
        X, y = self.construct_ml_dataset()
        
        X_train, X_test, y_train, y_test = self.get_test_train_split(X, y)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def generate_random_list(length, seed):
        np.random.seed(seed)
        rand_list = np.random.rand(length)
        normalized_list = rand_list / rand_list.sum()
        return normalized_list.tolist()

if __name__ == "__main__":
    data = pd.DataFrame(np.array([[1, "a"], [2, "b"], [3, "c"]]))

    dm = DensityMatrix(data)
    res = dm.main()

