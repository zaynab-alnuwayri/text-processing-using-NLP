# International Student Experience Analysis Using NLP

## Overview
This project analyzes Reddit posts from international students to identify common themes and experiences using Natural Language Processing (NLP) techniques. The analysis combines traditional topic modeling with modern transformer-based embeddings to uncover patterns related to academics, careers, finances, social belonging, and studying abroad.

## Research Question
What are the most common challenges, concerns, and experiences discussed by international students on Reddit?

## Dataset
The dataset consists of Reddit posts collected from discussions related to international student experiences. Posts were preprocessed and analyzed to extract meaningful topics and clusters.

## Methods

### 1. Latent Dirichlet Allocation (LDA)
- Text preprocessing and cleaning
- Tokenization and stop-word removal
- Topic modeling using Gensim LDA
- Coherence score evaluation to determine the optimal number of topics
- Topic visualization and interpretation

### 2. BERT-Based Text Analysis
- Document embeddings generated using a pre-trained BERT model
- K-Means clustering applied to document embeddings
- PCA visualization of document clusters
- Manual interpretation and labeling of clusters

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- spaCy
- Gensim
- Scikit-learn
- Transformers (BERT)
- Matplotlib

## Results

### LDA Topics
The optimal number of topics was determined using coherence scores. Topics revealed themes such as:
- Academic life
- Employment opportunities
- Study abroad experiences
- Social relationships
- Future planning

### BERT Clusters
BERT embeddings and K-Means clustering identified seven major discussion themes:
- Academic and University Life
- Social Belonging and Friendship
- Career and Job Concerns
- Financial Stress
- Studying Abroad and Adjustment
- Future Planning
- General International Student Experiences

## Visualizations
- Coherence score vs. number of topics
- PCA visualization of BERT clusters
- Distribution of posts across clusters


## Future Improvements
- Expand the dataset with additional Reddit communities
- Implement BERTopic for more advanced topic modeling
- Perform sentiment analysis within each topic cluster
- Compare multiple transformer models for clustering performance
