import pandas as pd
import numpy as np
import os


def load_data(fp):
    return pd.read_csv(fp, sep="\t")


def map_labels(df, fp_labels):
    df_labels = pd.read_csv(fp_labels, index_col=None, header=None)
    df_labels.columns = ["label"]
    df["label"] = df_labels["label"].values
    return df


def convert_na(df):
    df = df.fillna(0)
    return df


def remove_ipv6(df):
    df = df[df["ip.version"] != "6"]
    return df


def remove_unnecessary_columns(df, columns_drop):
    # columns=['ip.id']
    df = df.drop(columns=columns_drop)
    return df

def expand_hex_ip_columns(df, original_column, delim=":"):
    """expand into 1 column for each byte"""
    if original_column not in df.columns:
        print(
            "Failed expand_hex_ip_columns(): original_column not in df.columns. Returning input df."
        )
        return df

    first_non_zero_value = df[original_column].iloc[(df[original_column] != 0).idxmax()]
    num_hex_columns = len(first_non_zero_value.split(delim))
    hex_columns = [original_column + "_" + str(i) for i in range(num_hex_columns)]

    if delim == ":":
        df_split = df[original_column].str.split(":", expand=True).fillna("0")
        df[hex_columns] = df_split.map(lambda x: int(x if x != "" else "0", 16))
        df[hex_columns] = df[hex_columns].astype(float)
        # df.drop(original_column, axis=1, inplace=True)
    if delim == ".":
        df[hex_columns] = df[original_column].str.split(".", expand=True)
        df[hex_columns] = df[hex_columns].astype(float)
        # df.drop(original_column, axis=1, inplace=True)

    return df

def binary_encode_ports(df, port_columns):
    """encode ports into binary columns (expands fewer columns than one-hot.)"""
    num_bits = 16  # Number of bits to represent the port number

    for port in port_columns:
        df[port] = df[port].fillna(0).astype(int)
        df_port = pd.DataFrame(
            np.array([list(format(x, f"0{num_bits}b")) for x in df[port]]).astype(int),
            columns=[f"{port}_bit_{i}" for i in range(num_bits)],
        )
        df = pd.concat([df, df_port], axis=1)
        # df.drop(port, axis=1, inplace=True)

def write_data(df, fp_output):
    df.to_csv(fp_output, sep="\t")


if __name__ == "__main__":
    df = load_data("datasets/mirai/mirai_pcap.tsv")
    df = map_labels(df, "datasets/mirai/mirai_labels.csv")
    df = convert_na(df)
    df = remove_ipv6(df)

    df = expand_hex_ip_columns(df, 'eth.src', ':')
    df = expand_hex_ip_columns(df, 'eth.dst', ':')
    df = expand_hex_ip_columns(df, 'ip.src', '.')
    df = expand_hex_ip_columns(df, 'ip.dst', '.')

    object_columns = [col for col, dtype in zip(df.columns, df.dtypes.values) if dtype == 'object']
    df = remove_unnecessary_columns(df, object_columns)

    write_data(df, "datasets/mirai/mirai_processed.tsv")