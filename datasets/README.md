# Datasets

This directory contains the datasets used in this project. Please follow the instructions below to download the required files.


## Kitsune Mirai Dataset

The Kitsune Mirai dataset [1] is used for evaluating network intrusion detection systems.

You can download the dataset directly from the following link:
* [https://github.com/ymirsky/Kitsune-py/blob/master/mirai.zip](https://github.com/ymirsky/Kitsune-py/blob/master/mirai.zip)

Please download and unzip the `mirai.zip` file and place it in this directory.

To extract features from the resulting .pcap file, run the `DensityMatrixRobustness/data_preparation/ids/utils/feature_extraction.py` script via the command line: 

`python feature_extraction.py -i mirai.pcap -o mirai_pcap.csv`

The `feature_extraction.py` script requires a Wireshark installation: [https://www.wireshark.org/](https://www.wireshark.org/)

## UNSW-NB15 Dataset

The UNSW-NB15 dataset [2] is another benchmark dataset for network intrusion detection.

1.  **Download the Dataset**: You can find the dataset and its description on the official project page:
    * [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

2.  **Process the Data**: To convert the raw pcap files into a usable byte format, you will need the `Payload-Byte` [3] tool.
    * **Tool Repository**: [https://github.com/Yasir-ali-farrukh/Payload-Byte/tree/main](https://github.com/Yasir-ali-farrukh/Payload-Byte/tree/main)
    * Follow the instructions in the `Payload-Byte` repository to process the downloaded UNSW-NB15 data.


## Producing the Feature Datasets

Run the dataprocessing scripts in the following order:

```
data_preparation/ids/mirai.ipynb
data_preparation/ids/unsw-nb15.ipynb
encodings/ids/mirai.ipynb
encodings/ids/unsw-nb15.ipynb
```


---

[1] Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). Kitsune: an ensemble of autoencoders for online network intrusion detection. arXiv preprint arXiv:1802.09089.

[2] Moustafa, N., & Slay, J. (2015, November). UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). In 2015 military communications and information systems conference (MilCIS) (pp. 1-6). IEEE.

[3] Farrukh, Y. A., Khan, I., Wali, S., Bierbrauer, D., Pavlik, J. A., & Bastian, N. D. (2022, December). Payload-byte: A tool for extracting and labeling packet capture files of modern network intrusion detection datasets. In 2022 IEEE/ACM International Conference on Big Data Computing, Applications and Technologies (BDCAT) (pp. 58-67). IEEE.