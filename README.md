# Telegram Review Sentiment Analysis

## TRY IT HERE: https://sentimen-analysis-telegram.streamlit.app/

This project analyzes the sentiment of user reviews for the **Telegram app** from the Google Play Store.  
It scrapes reviews, preprocesses the text data, performs sentiment analysis using **lexicon-based, Random Forest, and LSTM models**, and visualizes the results.

---

## Project Description

This repository contains the code and data for a sentiment analysis project on Telegram user reviews.  
The primary goal is to classify reviews into **positive, negative, or neutral** categories.

The project is divided into two main parts:

1. **Web Scraping**  
   Retrieves user reviews of the Telegram app from the Google Play Store.

2. **Sentiment Analysis**  
   Cleans and preprocesses the review text and then applies different models to classify the sentiment.

---

## How it Works

The project uses a pipeline from **raw data → sentiment classification**:

1. **Data Collection**  
   - `scrap-review.ipynb` uses the `google-play-scraper` library.  

2. **Data Cleaning & Preprocessing** (in `sentimen_analysis.ipynb`)  
   - **Cleaning**: Removes special characters, URLs, and numbers.  
   - **Case Folding**: Converts all text to lowercase.  
   - **Slang Correction**: Replaces informal slang with standard Indonesian words.  
   - **Tokenization**: Splits text into tokens (words).  
   - **Stopword Removal**: Removes common words (e.g., *dan, di, dari*).  
   - **Sentiment Labeling**: Assigns sentiment via a **lexicon-based approach**.

3. **Model Training & Evaluation**  
   - Random Forest  
   - ANN (Artificial Neural Network)  
   - LSTM (Long Short-Term Memory)  

---

## Models and Scenarios

Three approaches were tested:

### 🔹 Scenario 1: Random Forest with TF-IDF
- **Model**: Random Forest (ensemble of decision trees).  
- **Features**: TF-IDF (200 most important features).  
- **Strength**: Simple and interpretable.

### 🔹 Scenario 2: ANN with TF-IDF
- **Model**: Feedforward Neural Network (ANN).  
- Includes **BatchNormalization** and **Dropout** layers.  
- **Features**: TF-IDF vectors.  
- **Strength**: Learns more complex patterns than Random Forest.

### 🔹 Scenario 3: LSTM with Word2Vec
- **Model**: Bidirectional LSTM.  
- **Features**: Word2Vec embeddings (dense vectors capturing semantics).  
- **Strength**: Best at capturing context and sequence in text.

---

## Results and Evaluation

**Word Clouds** show the most frequent words in **positive, negative, and neutral** reviews.  
They provide insight into the main themes of user feedback.

### Model Performance

| Scenario | Model          | Features    | Test Accuracy |
|----------|---------------|-------------|---------------|
| 1        | Random Forest | TF-IDF      | **71.9%**     |
| 2        | ANN           | TF-IDF      | **~73.9%**    |
| 3        | LSTM          | Word2Vec    | **89.2%**     |

✅ The **LSTM + Word2Vec** model performed best, significantly outperforming the others.

---

## Getting Started

### Prerequisites
- Python 3  
- Packages listed in `requirements.txt`

### Installation
Clone the repository:
```bash
git clone https://github.com/friendotjava/telegram-review-sentiment-analysis.git
cd telegram-review-sentiment-analysis
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebooks
1. **Scrape reviews**  
   Run `scrap-review.ipynb` → saves data as `telegram-review.csv`.

2. **Run sentiment analysis**  
   Run `sentimen_analysis.ipynb` → preprocess data & train models.

---

## Built With

- **[gensim](https://radimrehurek.com/gensim/)** → Word2Vec embeddings  
- **[google-play-scraper](https://pypi.org/project/google-play-scraper/)** → Review scraping  
- **[matplotlib](https://matplotlib.org/)** & **[seaborn](https://seaborn.pydata.org/)** → Visualization  
- **[nltk](https://www.nltk.org/)** & **[Sastrawi](https://pypi.org/project/Sastrawi/)** → NLP preprocessing (Indonesian)  
- **[pandas](https://pandas.pydata.org/)** & **[numpy](https://numpy.org/)** → Data manipulation  
- **[scikit-learn](https://scikit-learn.org/)** → Machine learning (Random Forest, TF-IDF)  
- **[TensorFlow](https://www.tensorflow.org/)** → ANN & LSTM models  
- **[wordcloud](https://github.com/amueller/word_cloud)** → Word cloud visualizations  
