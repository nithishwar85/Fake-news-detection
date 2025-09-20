Fake News Detection Project – Detailed Explanation

Overview
This project focuses on detecting fake news using machine learning and deep learning models. The dataset used is the Fake and Real News Dataset from Kaggle, containing two CSV files:

Fake.csv – containing fake news articles

True.csv – containing real news articles

The main goal is to classify news articles as either Fake or True using three approaches: Logistic Regression (TF-IDF features), LSTM (Deep Learning model with embeddings), and BERT (Transformer-based model).

Dataset Preparation
The datasets were labeled for classification, with Fake news as 0 and True news as 1. The two datasets were then combined into a single dataset and shuffled randomly to avoid ordering bias. The features (news text) and labels were split into training and testing sets with an 80-20 ratio.

Logistic Regression Model
The text data was transformed into numerical features using TF-IDF vectorization. Common English stopwords were removed, and words appearing in more than 70% of documents were ignored. Logistic Regression was trained on the TF-IDF features with a maximum of 1000 iterations to ensure convergence. The model achieved an accuracy of 98.5%. The confusion matrix showed precise classification of Fake and True news.

BERT Model (Transformer-based)
The BERT tokenizer converted news text into token IDs compatible with BERT. Sequences were padded or truncated to a maximum length of 256. TensorFlow datasets were created for training and testing with a batch size of 8. The pre-trained BERT-base uncased model was fine-tuned for two epochs. The model achieved perfect accuracy on the demonstration dataset.

LSTM Model (Deep Learning)
Text data was tokenized and converted into sequences of integers, padded to a maximum length of 300. The model architecture included an embedding layer to convert word indices into dense vectors, an LSTM layer to capture sequential dependencies, and a dense layer with sigmoid activation to output probabilities. The model was trained for 3 epochs with a batch size of 64 and achieved an accuracy of 99%.

Results Summary Model Accuracy Logistic Regression 98.5% LSTM (Deep Learning) 99% BERT (Transformer) 100%
All models achieved high accuracy. Logistic Regression is fast and interpretable, LSTM captures sequential dependencies, and BERT leverages large-scale pre-training for superior performance.

Visualization
A confusion matrix was generated for Logistic Regression, showing the number of Fake and True news articles correctly and incorrectly classified. This visualization helped evaluate the model’s performance more intuitively.

Conclusion
Multiple models can effectively detect fake news. Deep learning models such as LSTM and BERT achieve near-perfect performance. This workflow can be extended to other text classification tasks. The project combines data preprocessing, feature extraction, traditional machine learning and deep learning approaches, evaluation, and visualization.

Attachments for Word Document or Website
Include the following in your Word document or website:

Dataset description

Model summaries

Accuracy tables

Confusion matrix heatmaps

Sample predictions
