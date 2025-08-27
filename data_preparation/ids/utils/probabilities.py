from pennylane import numpy as np
import pandas as pd

from utils.flow_windows import Flows

class Probabilities:
    def __init__(self, df_flows, bases, base=None, method="bayesian"):
        """
        Initialize the Probabilities object.

        Parameters:
        df_flows (DataFrame): A DataFrame containing the flow data.
        bases (list): A list of strings representing the base columns for probability calculations.
        method (str): The method to use for probability calculations; defaults to 'bayesian'.
        """
        self.df = df_flows
        self.bases = bases
        if base is None or base not in self.bases:
            self.base = self.bases[0]
        else:
            self.base = base
        self.method = method
        self.global_probs = None
        self.local_probs = None

    def set_bases(self, bases):
        """
        Set the bases for probability calculations if they exist within the DataFrame columns.

        Parameters:
        bases (list): A list of strings representing the new base columns for probability calculations.
        """
        if all(base in self.df.columns for base in bases):
            self.bases = bases
        else:
            print("Invalid 'bases': provided bases are not found within df.columns")

    def set_base(self, idx):
        """
        Set the base for probability calculations based on the index provided.

        Parameters:
        idx (int): The index of the base in the bases list to be used for calculations.
        """
        if idx < len(self.bases):
            self.base = self.bases[idx]
        else:
            print("Invalid 'base': index out of bounds of bases range")

    @staticmethod
    def filter_nans_and_zeros(df, base):
        """
        Filter out NaN and zero values from the DataFrame for a given base.

        Parameters:
        df (DataFrame): The DataFrame to filter.
        base (str): The column name based on which the filtering is applied.

        Returns:
        DataFrame: The filtered DataFrame.
        """
        return df[df[base].notna() & (df[base] != 0)]

    def get_base_value_counts(self, df, base):
        """
        Get the normalized value counts for a given base from the DataFrame.

        Parameters:
        df (DataFrame): The DataFrame from which to calculate value counts.
        base (str): The column name based on which the value counts are calculated.

        Returns:
        Series: A Series containing normalized value counts for the base.
        """
        df = self.filter_nans_and_zeros(df, base)
        return df[base].value_counts(normalize=True, dropna=False)

    def get_global_probabilities(self):
        """
        Calculate and store global probabilities for each base in the bases list.
        """
        self.global_probs = {
            base: self.get_base_value_counts(self.df, base) for base in self.bases
        }

    def get_local_probabilities(self):
        """
        Calculate and store local probabilities for each base within each flow window.
        """
        local_probabilities = {}
        for flow in self.df["flow_window"].unique():
            #print(f"{flow}/{len(self.df['flow_window'].unique())}")
            df_flow = self.df[self.df["flow_window"] == flow]
            local_probabilities[flow] = {}
            for base in self.bases:
                local_probabilities[flow][base] = self.get_base_value_counts(
                    df_flow, base
                )
        self.local_probs = local_probabilities

    def bayesian_update(self, global_probs, local_probs):
        """
        Perform a Bayesian update to calculate the posterior probabilities.

        Parameters:
        global_probs (Series): The global probabilities.
        local_probs (Series): The local probabilities for a specific flow window.

        Returns:
        dict: A dictionary containing the updated posterior probabilities.
        """
        evidence = sum(
            global_probs.get(label, 0) * local_probs.get(label, 0)
            for label in global_probs.index.union(local_probs.index)
        )
        posteriors = {}
        for label in global_probs.index.union(local_probs.index):
            prior = global_probs.get(label, 0)
            likelihood = local_probs.get(label, 0)
            posterior = (prior * likelihood) / evidence if evidence > 0 else 0
            posteriors[label] = posterior

        return posteriors

    def get_weighted_probabilties(self):
        """
        Apply weighted probabilities to each row based on global and local probabilities.
        """
        method_name = self.method + "_update"
        method = getattr(self, method_name)

        def apply_probs(row):
            flow = row["flow_window"]
            ph = row[self.base]

            probs = method(
                self.global_probs[self.base], self.local_probs[flow][self.base]
            )

            if ph == 0:
                return 0
            else:
                return probs[ph]

        self.df["dm_prob"] = self.df.apply(apply_probs, axis=1)

    def get_softmax_probabilties(self):
        def softmax(x):
            e_x = np.exp(x - np.max(x))  # subtract max to stabilize
            return e_x / e_x.sum(axis=0)

        for flow in self.df['flow_window'].unique():
            self.df.loc[self.df['flow_window'] == flow, 'dm_prob_softmax'] = softmax(self.df[self.df['flow_window'] == flow]['dm_prob'].values)

    def main(self):
        self.get_global_probabilities()
        self.get_local_probabilities()
        self.get_weighted_probabilties()
        self.get_softmax_probabilties()
        return self.df


if __name__ == "__main__":
    bases = ["ip.proto", "tcp.dstport", "udp.dstport"]
    data = pd.DataFrame(np.array([[1, "a"], [2, "b"], [3, "c"]]))
    df_flows = Flows(data)

    probs = Probabilities(df_flows, bases, None, "bayesian")
    res = probs.main()
