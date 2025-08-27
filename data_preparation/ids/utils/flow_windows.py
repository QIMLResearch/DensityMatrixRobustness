from pennylane import numpy as np
import pandas as pd
from datetime import datetime


class Flows:
    """
    A class to represent and process network flows based on a pandas DataFrame.

    Attributes
    ----------
    df : pandas.DataFrame
        A DataFrame containing the flow data.
    method : str
        The method used to create flows. Currently only 'time' method is implemented.
    flow_duration : int
        The duration of each flow in seconds.
    flows : NoneType
        Placeholder for storing the flows, not yet implemented.

    Methods
    -------
    assign_datetime_column():
        Converts epoch time to datetime and sorts the DataFrame by this new datetime column.

    get_flows_using_time_window():
        Assigns each row in the DataFrame to a flow window based on 'frame.date_time' column.

    main():
        The main method to be called externally. It runs necessary preprocessing to assign flows.
    """

    def __init__(self, df, method="time", flow_duration=1):
        """
        Constructs all the necessary attributes for the Flows object.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame containing the flow data.
        method : str, optional
            The method used to create flows (default is "time").
        flow_duration : int, optional
            The duration of each flow in seconds (default is 1 second).
        """
        self.df = df
        self.method = method
        self.flow_duration = flow_duration  # in seconds
        self.flows = None

    def assign_datetime_column(self):
        """
        Converts epoch times in 'frame.time_epoch' column to datetime objects and
        sorts the DataFrame by this new 'frame.date_time' column.
        """
        self.df["frame.date_time"] = self.df["frame.time_epoch"].apply(
            lambda x: datetime.fromtimestamp(x)
        )
        self.df = self.df.sort_values(by="frame.date_time").reset_index(drop=True)

    def get_flows_using_time_window(self):
        """
        Assigns each row in the DataFrame to a flow window based on 'frame.date_time' column.

        Flow windows are determined by dividing the time range into intervals of 'flow_duration'.
        Each row is then categorized into these intervals.
        """

        flow_td = pd.Timedelta(f"{self.flow_duration}S")

        # Calculate the min and max time, adjusting for the window size
        min_time = self.df["frame.date_time"].min() - flow_td
        max_time = self.df["frame.date_time"].max() + flow_td

        # Create the range of bins
        time_range_extended = pd.date_range(start=min_time, end=max_time, freq=flow_td)

        # Use pd.cut to assign each timestamp to a time window
        self.df["flow_window"] = pd.cut(
            self.df["frame.date_time"],
            bins=time_range_extended,
            right=False,
            labels=False,
        )

    def main(self):
        """
        The main method that orchestrates the flow assignment process.

        It calls the methods to convert epoch times to datetime and assign time windows,
        effectively categorizing the DataFrame rows into flows based on time.

        Returns
        -------
        pandas.DataFrame
            The updated DataFrame with a new 'flow_window' column indicating the flow each row belongs to.
        """
        self.assign_datetime_column()
        self.get_flows_using_time_window()
        return self.df


if __name__ == "__main__":
    data = pd.DataFrame(np.array([[1, "a"], [2, "b"], [3, "c"]]))

    flows = Flows(data)
    res = flows.main()
