# Quora Question Similarity Detector

## Introduction
This project is the outcome of the INSY669 Text Analytics course at McGill University. The team, dedicated to enhancing the Quora user experience by identifying duplicate questions, comprises:

- Avi Malhotra
- Krishan Gupta
- Nandani Yadav
- Om Sangwan
- Yash Josi

Utilizing advanced NLP techniques and machine learning, we aimed to streamline content quality and improve platform efficiency.

## Dataset
We utilized the Quora Question Pairs dataset, featuring over 400,000 question pairs with binary labels indicating duplicate status.

## Exploratory Data Analysis (EDA)
Key insights from our EDA included class imbalance and the prevalence of stopwords. We analyzed question lengths and word frequencies to understand the data distribution.

## Methodology
The project encompassed several stages:
- **Data Preprocessing**: Cleansing texts, interpreting LaTeX, expanding contractions, and removing stopwords.
- **Feature Engineering**: Deriving POS tags, quantifying word commonality, and calculating question length and word count disparities.
- **Vectorization**: Applying TFIDF and BERT for semantic embeddings.
- **Dimensionality Reduction**: Using SVD on TFIDF vectors to reduce sparsity.
- **Model Selection**: Testing various models and selecting the Gradient Boosting Classifier for its superior performance.

## Impact and Broader Applications
Beyond Quora, our model can improve bot detection, prevent plagiarism, and enhance customer support chatbots, among other applications.

## Conclusion
Our supervised learning model effectively addresses Quora's duplicate question problem, paving the way for future exploration of unsupervised learning methods to handle unseen data.

## Deployment
Visit our interactive Streamlit application to see the model in action: [Quora Question Similarity Detector](https://quora-question-similarity-detector.streamlit.app/).
