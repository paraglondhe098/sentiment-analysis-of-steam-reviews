# Steam Game Reviews Sentiment Analysis with LSTM Neural Networks

## Introduction

This project focuses on sentiment analysis of Steam game reviews using LSTM (Long Short-Term Memory) neural networks and word embeddings.
The goal of the project is to predict whether a review expresses a positive sentiment (1) or a negative sentiment (0) based on the text content of the reviews.

## Dataset

The dataset used in this project consists of Steam game reviews collected from Kaggle.
Each review is labeled with its corresponding sentiment, either positive or negative.
The dataset contains many reviews from various games, making it suitable for training a sentiment analysis model.\
Link - https://www.kaggle.com/datasets/andrewmvd/steam-reviews?select=dataset.csv

## Requirements

Listed in requirements.txt

You can install the required libraries using pip with the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

The project repository is organized as follows:

```
├── data/
│   └── reviews.csv          # CSV file containing the Steam game reviews dataset
│
├── notebook
│   └── Sentiment-analysis-Steam-Reviews.ipynb       # Jupyter notebook containing the main code and analysis
│
├── models/
│   └── c1_lstm_model_acc_highaccr0.827.keras        # Pre-trained LSTM model for sentiment analysis
│
├── main
│   └── main.py     # Python script for text preprocessing and word embeddings and displaying results
│
├── README.md                # Project documentation (you are here)
│
└── requirements.txt         # List of project dependencies
```

## Running the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/steam-reviews-sentiment-analysis.git
cd steam-reviews-sentiment-analysis
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset and place it in the `data/` directory.

4. Open the `Sentiment-analysis-Steam-Reviews.ipynb` notebook using Jupyter Notebook or Jupyter Lab to explore the data, train the LSTM model, and perform sentiment analysis on the Steam game reviews.

## Model Performance

The LSTM neural network has been trained on the dataset, and the pre-trained model `c1_lstm_model_acc_highaccr0.827.keras` is provided in the `models/` directory. The model achieved an accuracy of 82.7% on the test set and performed well in predicting sentiment labels for Steam game reviews.

## Conclusion

This project demonstrates the application of LSTM neural networks and word embeddings in sentiment analysis of Steam game reviews. The trained model can effectively predict whether a review expresses a positive or negative sentiment. Feel free to use, modify, and expand upon this project for further research and analysis.

If you have any questions or suggestions, please feel free to contact the project contributors.

Happy analyzing!
