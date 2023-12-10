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



Proposed File Structure
project-root/
│
├── data/
│   ├── dataset1/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── ...
│   ├── dataset2/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── ...
│   ├── dataset3/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── ...
│   ├── dataset4/
│       ├── raw/
│       ├── processed/
│       └── ...
│
├── src/
│   ├── algorithms/
│   │   ├── algorithm1/
│   │   ├── algorithm2/
│   │   ├── algorithm3/
│   │   └── algorithm4/
│   ├── utils/
│   └── main.py
│
├── results/
│   ├── algorithm1/
│   ├── algorithm2/
│   ├── algorithm3/
│   └── algorithm4/
│
├── notebooks/
│   ├── exploration.ipynb
│   └── visualization.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
