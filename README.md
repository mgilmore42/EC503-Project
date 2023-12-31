# EC503-Project

## Installation

In order to use this repository you first need to install the required packages.
This can be done by running the following command:

```bash
pip install -r requirements.txt
```

## Downloading the data

All data is stored on Kaggle and can be downloaded using the Kaggle API.
Simply ensure your [Kaggle API key](https://www.kaggle.com/docs/api#authentication) is downloaded and stored in the `~/.kaggle/` directory.
Then, run the following commands:

```bash
python main.py download
```

This generates the `data/` directory and downloads the data into it.

## Training the models

After downloading the data, you can train the models by running the following command:

```bash
python main.py train
```

Afterwards all the results will be stored in the `results/` directory.