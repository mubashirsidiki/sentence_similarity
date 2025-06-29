# Sentence Similarity Analysis Project

This project demonstrates two different approaches to measuring sentence similarity using modern machine learning techniques. The project includes both a **Transformer-based approach** and a **Siamese Neural Network approach** to understand how similar two pieces of text are to each other.

## 🎬 Original Use Case: Rotten Tomatoes Movie Review Analysis

**This project originated from a unique and innovative idea: comparing audience vs. critic consensus on movies from Rotten Tomatoes.** 

As a movie enthusiast active in movie communities, I noticed recurring debates about the differences between audience and critic opinions on platforms like Rotten Tomatoes. Sometimes their opinions align, but other times they clash, sparking heated discussions on platforms like Reddit and Facebook.

### 🎯 The Original Research Question
**"Do critics and audiences really have such different opinions about movies, or are we overstating their differences?"**

This led to the creation of a custom dataset by scraping Rotten Tomatoes data for top box office movies of 2024, comparing audience consensus vs. critic consensus to measure their actual similarity using advanced NLP techniques.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [🎬 Original Rotten Tomatoes Use Case](#-original-rotten-tomatoes-use-case)
- [Approach 1: Transformer Neural Network](#approach-1-transformer-neural-network)
- [Approach 2: Siamese Neural Network](#approach-2-siamese-neural-network)
- [Datasets](#datasets)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results Comparison](#results-comparison)
- [Key Insights](#key-insights)

## 🎯 Project Overview

Sentence similarity is a fundamental task in natural language processing that measures how similar two pieces of text are in meaning. This project explores two different methodologies:

1. **Transformer-based Approach**: Uses pre-trained language models to generate sentence embeddings
2. **Siamese Neural Network Approach**: Trains a custom neural network to learn similarity patterns

Both approaches are applied to real-world scenarios including movie review analysis and general sentence comparison.

## 🎬 Original Rotten Tomatoes Use Case

### The Inspiration Behind the Project

Being a movie enthusiast and active in movie communities, I've noticed a recurring debate on platforms like Rotten Tomatoes. This debate revolves around the different opinions of audiences and movie critics. Sometimes, their opinions align, but other times, they clash, sparking heated discussions on platforms like Reddit and Facebook.

### Examples of the Debate:

**The Super Mario Bros. Movie Controversy:**
- Critics gave it low scores
- Audiences loved it
- This caused quite a stir in the community

**Ant-Man and The Wasp (2018):**
- Critics had lukewarm reception
- Audiences enjoyed it more
- Led to memes and discussions about critic vs. audience disconnect

### The Original Research Question

**"Are critics and audiences really that different, or are we overstating their differences?"**

This question led to the creation of a custom dataset by manually scraping Rotten Tomatoes data for the top box office movies of 2024.

### Data Collection Process

I decided to gather data from [Rotten Tomatoes](https://www.rottentomatoes.com/browse/movies_in_theaters/sort:top_box_office), focusing on the top box office movies of 2024. The data includes:
- Movie names
- Audience consensus
- Critic consensus

**Data Collection Method:**
- Used a combination of web scraping and manual data entry
- Replaced specific nouns with pronouns for consistency (e.g., "Kung Fu Panda is great" → "It is great")
- Carefully cleaned and processed the collected data

### The Innovation

This approach is unique because:
1. **Original Dataset**: Created from scratch by scraping real movie review data
2. **Real-world Application**: Addresses an actual debate in movie communities
3. **Quantitative Analysis**: Uses NLP to measure similarity rather than just qualitative comparison
4. **Community Relevance**: Tackles a question that movie fans actually care about

## 🚀 Approach 1: Transformer Neural Network

### What is it?
This approach uses the **BGE-M3** transformer model, a state-of-the-art sentence embedding model that converts text into numerical vectors (embeddings) that capture semantic meaning.

### How it works:
1. **Model Selection**: Uses BGE-M3, ranked #2 on the MTEB (Massive Text Embedding Benchmark) leaderboard
2. **Sentence Embedding**: Converts sentences into 1024-dimensional vectors
3. **Similarity Calculation**: Uses cosine similarity to measure how similar two vectors are
4. **Classification**: Categorizes similarity into "Strong" (≥0.7), "Moderate" (≥0.5), or "Weak" (<0.5)

### Key Features:
- **Pre-trained Model**: Leverages BGE-M3, trained on massive text datasets
- **Cosine Similarity**: Measures angle between vectors (range: -1 to 1)
- **Real-world Application**: Analyzes movie critic vs. audience consensus
- **GPT Comparison**: Validates results against OpenAI's GPT model

### Example Usage:
```python
# Compute similarity between two sentences
similarity = compute_similarity("I had a bad day", "Everything was terrible today")
# Returns: 0.797 (Strong similarity)
```

### Results:
- Successfully identifies semantic similarity between movie reviews
- Achieves good agreement with GPT model classifications
- Handles nuanced language differences effectively

## 🧠 Approach 2: Siamese Neural Network

### What is it?
A custom neural network architecture that learns to compare sentence pairs by training on labeled examples. Uses bidirectional LSTM layers to understand sentence context.

### How it works:
1. **Word Embeddings**: Converts words to 50-dimensional vectors using Word2Vec
2. **Siamese Architecture**: Two identical LSTM networks process each sentence
3. **Feature Merging**: Combines sentence representations with additional features
4. **Similarity Prediction**: Outputs a probability score (0-1) for similarity

### Architecture Details:
- **Embedding Dimension**: 50
- **LSTM Units**: 50 (bidirectional)
- **Dense Units**: 50
- **Dropout Rates**: 17% (LSTM), 25% (Dense)
- **Activation**: ReLU

### Key Features:
- **Custom Training**: Learns from scratch on sentence pair dataset
- **Bidirectional LSTM**: Captures context from both directions
- **Feature Engineering**: Includes word overlap statistics
- **Binary Classification**: Predicts similar (1) or dissimilar (0)

### Example Usage:
```python
# Compute similarity using trained model
similarity = compute_similarity("I had a bad day", "Everything was terrible today")
# Returns: 0.500 (Moderate similarity)
```

### Current Limitations:
- **Small Dataset**: Only 193 training pairs (needs 1,000+ for deep learning)
- **Poor Performance**: ~47% validation accuracy (close to random)
- **Early Stopping**: Model stops after 4 epochs due to overfitting

## 📊 Datasets

### 1. 🎬 Movie Reviews Dataset (`audience_vs_critic_pronoun.csv`) - **ORIGINAL DATASET**
- **Source**: **Manually scraped from Rotten Tomatoes** (top box office movies 2024)
- **Content**: Movie names, audience consensus, critic consensus
- **Size**: 17 movie entries
- **Use Case**: **Primary use case - Transformer approach analysis**
- **Innovation**: **Original dataset created from scratch to answer real community questions**

### 2. Sentence Pairs Dataset (`sentences.csv`)
- **Source**: Generated by ChatGPT
- **Content**: Sentence pairs with binary similarity labels
- **Size**: 193 sentence pairs
- **Use Case**: Siamese network training

### 3. Final Dataset (`final.csv`)
- **Content**: Processed movie review data
- **Size**: 17 entries
- **Use Case**: Analysis and visualization

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Core dependencies
pip install sentence-transformers matplotlib python-dotenv openai seaborn pandas numpy

# For Siamese network
pip install keras pandas gensim tensorflow tensorboard
```

### Environment Setup
1. Clone the repository
2. Install dependencies
3. Set up OpenAI API key (for GPT comparison)
4. Download datasets to `dataset/` folder

## 📖 Usage

### Running the Transformer Approach (Rotten Tomatoes Analysis)
```python
# Load the notebook: "1) Similarity Using Transformer Neural Network.ipynb"
# Follow the cells to:
# 1. Load BGE-M3 model
# 2. Process movie review data (your original Rotten Tomatoes dataset)
# 3. Compute similarity scores between audience and critic consensus
# 4. Compare with GPT results
# 5. Analyze the fascinating discoveries about critic vs. audience opinions
```

### Running the Siamese Network Approach
```python
# Load the notebook: "2) Similarity Using Siamese Neural Network.ipynb"
# Follow the cells to:
# 1. Prepare training data
# 2. Train the model
# 3. Evaluate performance
# 4. Test on new sentences
```

## 📈 Results Comparison

| Aspect | Transformer Approach | Siamese Network |
|--------|---------------------|-----------------|
| **Model Type** | Pre-trained BGE-M3 | Custom BiLSTM |
| **Training Data** | None (zero-shot) | 193 sentence pairs |
| **Performance** | High accuracy | ~47% validation accuracy |
| **Speed** | Fast inference | Slower training + inference |
| **Flexibility** | Limited to embedding similarity | Can be fine-tuned |
| **Resource Usage** | Moderate | High (training required) |

### Similarity Score Examples:

**Transformer Approach:**
- "I had a bad day" vs "I had so much fun" → 0.638 (Moderate)
- "I had a bad day" vs "Everything was terrible today" → 0.798 (Strong)

**Siamese Network:**
- "I had a bad day" vs "I had so much fun" → 0.492 (Weak)
- "I had a bad day" vs "Everything was terrible today" → 0.500 (Weak)

## 🔍 Key Insights

### 🎬 Fascinating Discoveries from Rotten Tomatoes Analysis:

**The Big Surprise: Critics and audiences aren't as different as we think!**

Our dive into the top-grossing movies of 2024 uncovered something interesting: critics and audiences aren't as different as we might think. Despite the hype around movies like "Mario" and "Ant-Man and the Wasp," the analysis showed that people's opinions can vary widely, but there's often more common ground than expected.

**Key Findings:**
- Critics and audiences often have similar opinions
- The transformer model effectively captures nuanced differences
- GPT model agrees with cosine similarity classifications
- Movie genre influences consensus similarity
- **Most importantly: Critics are just like us—they have similar views most of the time**

### Transformer Approach Strengths:
- **Immediate Use**: No training required
- **High Quality**: State-of-the-art performance
- **Scalable**: Works on any text without retraining
- **Robust**: Handles diverse language patterns
- **Perfect for Rotten Tomatoes Analysis**: Successfully measures critic vs. audience similarity

### Siamese Network Insights:
- **Data Dependency**: Requires substantial labeled data
- **Learning Potential**: Can be customized for specific domains
- **Architecture Flexibility**: Can be modified for different tasks
- **Training Challenges**: Needs careful hyperparameter tuning

## 🚀 Future Improvements

### For Rotten Tomatoes Analysis:
- **Expand Dataset**: Scrape more movies across different years and genres
- **Temporal Analysis**: Compare critic vs. audience opinions over time
- **Genre-specific Analysis**: Analyze differences by movie genre
- **Sentiment Analysis**: Add sentiment scoring to complement similarity
- **Interactive Dashboard**: Create a web app for real-time analysis

### For Transformer Approach:
- Try different embedding models (e.g., Sentence-BERT, MPNet)
- Implement semantic search capabilities
- Add multilingual support
- Create interactive similarity demo

### For Siamese Network:
- Collect larger training dataset (1,000+ pairs)
- Use pre-trained word embeddings (GloVe, Word2Vec)
- Implement data augmentation techniques
- Optimize hyperparameters systematically
- Add attention mechanisms

## 📝 Conclusion

This project demonstrates two complementary approaches to sentence similarity, with a special focus on the **original Rotten Tomatoes use case**:

1. **Transformer-based methods** are ready-to-use, high-performance solutions suitable for production applications
2. **Siamese networks** offer customization potential but require substantial data and training effort

### 🎬 The Rotten Tomatoes Revelation

The most exciting discovery from this project is that **critics and audiences aren't as different as we think**. Despite the heated debates in movie communities, our quantitative analysis using advanced NLP techniques shows that there's often more common ground than expected.

The transformer approach proves more practical for immediate use, while the Siamese network shows the potential for domain-specific customization with adequate data.

Both approaches contribute to our understanding of how machines can comprehend and compare human language, with applications ranging from content recommendation to automated text analysis.

**This project started with a simple question about movie opinions and evolved into a comprehensive exploration of sentence similarity using cutting-edge AI techniques.** 