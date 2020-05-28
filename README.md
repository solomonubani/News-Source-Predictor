# Using Deep Learning to predict the source of news
## Abstract
In this project, we present an approach and a model for predicting the source of a news article by taking advantage of the perceived partisan bias and sentiment of news sources. Our system developed the implemented model using a dataset we built by using a web scraping bot that crawled both CNN and The New York Posts. The dataset consists of a CSV file that contains over seven thousand news articles as well as their source--CNN or The New York posts. We created the news source prediction system by leveraging BERT sentence embeddings. After creating, training, and using the models to make predictions, we achieved an F1 score of about 89%.

## INTRODUCTION
![](https://d.newsweek.com/en/full/607719/djt.webp?w=790&f=082fb7486866b8a7efcddfcfb30cf017)

According to an article written by Gallup, During the 2016 presidential campaign, President Donald Trump and the GOP often said Hillary Clinton received favorable -- even preferential -- news coverage, while Democrats lamented that the media aired Trump's speeches and events uninterrupted. Overall, around 62% of Americans believe that the media favor one party over the other is at the highest it has been in more than 20 years.

**Article 1**: Why Democrats will come to regret impeaching President Trump

**Article 2**: Trump can't be trusted to make this decision

Can you guess which of these articles about President Trump was written by CNN and which was written by NYPost based on their article text? Most likely, yes you can. Perusing opinion articles about USA President Donald Trump written by CNN and The New York post, one can notice that there is some form of bias in the way different news outlets write about him. Some are usually full of praises for him while others not so much. 
Can we take advantage of this perceived partisan bias and sentiment to predict the source of a news article? In this project, we built a Machine/Deep Learning model that predicts the source of a news article by taking advantage of the perceived political partisan bias of news outlets. The model tries to understand this bias by doing a sentiment analysis and learning from thousands of news articles written by a particular news source. 

![](https://technologypursuit.edublogs.org/files/2019/05/Media-Bias-Chart_4.0_8_28_2018-min-1024x791-2e8iebg-1gop9db.jpg)

Fig 1. Infographic showing the partisan bias of news outlets
In figure 1, we see an infographic showing the perceived partisan bias of major news outlets. 

## 1.1	Technologies and Libraries
We used the following technologies in our project; Python as our programming language. Google collab as the Notebook for running and compiling our recommendation system. Anaconda was used to install libraries in the case of their absence. We used the data packages Numpy and Pandas for data manipulation. For charting and visualization, we used Matplotlib and Seaborn. For sentence embeddings, We used the BERT models from the Keras and Tensorflow library. For crawling the news websites, we used Selenium. Lastly, we used Github for version control.
## 2.0	METHODOLOGY
### 2.1	Dataset:
We could not find any existing dataset for this project. We needed data mostly about President Trump and not commentary news but mostly opinion news articles. We settled for two news outlets CNN and NYPost. We resolved to build our own dataset by building a web scraping bot that crawled both sites using Selenium and put the data into a CSV file. The CSV file has two columns. The article column and the source column.  The article column contains text for the news article and the sentiment column contains source of the news. The source column can have two values i.e. 0 for CNN and 1 for NYPost which makes our problem a binary classification problem. Graph 1 belows shows the count of the news articles for both news sources considered in this project.
![](https://lh4.googleusercontent.com/QBMC_w8KOsQOkAUahRLLYG1EDz7UJxxfizA86QRF-03Ne9_D-s7vZ3dL2pAoyta5zORMFpYDgCjNehApTcI5BZqFwiGPubwWWOpCBweaGQYGMGVLiunVDpJFT-sF9PoOtRkxVsOV)

Graph 1.Dataset showing count of articles for both sources

0: CNN: 6145 News articles

1: NYPost: 1362 News articles
![](https://lh4.googleusercontent.com/Ad5ZWw5Vpqkz0c_C3DhDXWQIF78vG-rTCRLeOTQkX1LRTIgHiLqF9W-nKT0TdA2xs6t5c3lqqkrtMq-QCCw2qfmdRAJoqSUcV5PIAOE7DYGM82KYdrASS3aUnGX2Q6duwlYWLPPI)

Table 1.Dataset showing 1 entry
### 2.2	Data Preprocessing
Our dataset contained punctuations and HTML tags. In this section, we describe how we performed preprocessing on the strings to remove special characters, multiple tags white spaces, single characters,  and HTML tags from the string. We also removed every mention of ‘CNN’ from the news articles as it would become trivial to predict the source of a news article if ‘CNN’ is a word in it.
We divided our dataset into train and test sets. The train set was used to train our deep learning model while the test set was used to evaluate how well our model performs. We randomly divided the dataset into 80% for the training set and 20% for the testing set.

### 2.3	Data exploration
In this step, we performed an Exploratory Data Analysis (EDA). We did this to examine the data we are working with and see if we can observe any noticeable trends from the data which we can take into account when implementing our system. 

![](https://lh5.googleusercontent.com/B1pSESN6zJr5Dp_DJdFHIk5F20LmCvwI-kSpr6_8UVqu7XmdGuRXhj43bm1MnaN9ENMoavxq8zKouhNE93e_R746HZDhfmHoyYB0dn9T)

Graph 2: Graph showing clusters of articles from both news sources using BERT sentence embeddings as features

![](https://lh3.googleusercontent.com/ALZOJBJVXrJgzQyPE-D7v0IUt1kiVhEsMAHymBYv3RRinTAv9d4uDbXfWLnZfhcD4Elnr0dzln5zGAf-3LCircM9U8R3il0W5vUfVft1i9Qnt14AQ6JhB8-spZgdfZKwFVrHifav)
Graph 3: Graph showing most common words in the dataset

Graph 3 depicts the count of the most common words in the data set. We can see from the graph that the word ‘president’ is the most frequently used word with a total count of about 12,000 while ‘war’ is one of the least frequently used words with only 3,000 counts.
##  3.0	MODEL ARCHITECTURE
### 3.1	Embeddings - What are Word Embeddings? 
Machine Learning algorithms and almost all Deep Learning Architectures are incapable of processing strings or plain text in their raw form. They require numbers as inputs to perform any sort of job, be it classification, regression etc. in broad terms. And with the huge amount of data that is present in the text format, it is imperative to extract knowledge out of it and build applications.
A Word Embedding format generally tries to map a word using a dictionary to a vector. In this project, we use a special type of embedding called Sentence Embeddings. Sentence embeddings embed a full sentence (rather than just word) into a vector space. 
![](https://lh4.googleusercontent.com/7bCUYDwIG1CaKcczvG6VDK392oELUIStOnn3kmYZXD9IQZXeZ3eqggKLrvGapZnIewAufyA7wTE7FIcKAF0orj_-kmmWoixMakNgktSwYmJbWukPifhOfq6NfoEw0I456M3mCTGf)

Figure 2: sentence embedding using BERT for extractive summarization
In this project, to generate Sentence Embeddings, we use BERT, an already trained Sentence Transformer model to embed sentences. 
### 3.2	Model Layers
A Sequential Neural network model was used with two layers:
Layer 1: This Hidden layer is a 16 unit dense network with an input shape set to the embedding size (768). It uses the relu activation function.  
Layer 2: The Output layer comprises a layer dense network with sigmoid activation.
![](https://lh6.googleusercontent.com/wGelZyoMHs9jb5OKHdv9l6dYOhZF1edAa2DsX8bMCgMwUq-CXDHAURaH0QbrgrqVikwzzO5eBgl_r0JuTtHSN4QgetTgCvq2IANbCzDDDcsUnpt-DfH_bwJMN7oFo8BDqz1rBwy-)

Table 2: Model summary
### 3.3	Model Parameters
**Activation Function:** We used ReLU as the activation function. ReLU is a non-linear activation function, which helps complex relationships in the data to be captured by the model. Sigmoid activation function was used in the output layer.

**Optimizer:** We use adam optimizer, which is an adaptive learning rate optimizer.

**Loss Function:** We will train a network to output a probability over the 2 classes using binary Cross-Entropy loss. It is very useful for multi-class classification.

**Class Weights:** Because we have an imbalanced data set, CNN(6k+) versus NYPost(1k+) we added weights to the Neural Network model. We assigned the NYPost class a weight 6 times that of the CNN class

We ran the model with Batch Size of 500 and an Epoch of 20. 

## 4.0 RESULTS AND DISCUSSION
We used three metrics to evaluate our news source prediction system: F1 score, precision, and recall. Precision is the proportion of correct news article sources in the list of all the news article sources returned by the model while recall is the ratio of the correct news article sources returned by the model to the total number of the correct news article sources that could have been returned and the F1 score is the harmonic mean of precision and recall. The formula for recall and precision are: 

![](https://miro.medium.com/max/888/1*7J08ekAwupLBegeUI8muHA.png)
![](https://miro.medium.com/max/1124/1*V27Dd47fxtCB-u561I8nYQ.png)

![](https://lh6.googleusercontent.com/_fE1Rz4B3al22Jus_wNEOMH0VzRn9kc-Wsqw-I7IBkljz6uCH8Gu1W2rLZzLe3ezXaw3wwY2gz1ghvhpxbzD-H6tym02SCs_ULR0o9wFisdhvoCeAxqZ-onmyeaGX3INGjbv85EV)

Graph 4: Graphs showing Training and validation loss and accuracy

![](https://lh6.googleusercontent.com/Cf5fm8dUmNk2lnC7sW-ZzsFUsFuRlnON7nsZ79e7kGw92fz14aoKtYS2I1XvVqNHI8QA0I9fnAIpzYEnko_oGwZ2CKWvxhdTbnME3nGy03Sp0C43qLu4StBeXA6bwH7rLWGcdcko)

Table 3: Table showing the metrics of the model. Precision, recall, F1-score.
The recall is 89%, the precision is 91% and the F1 score is 89%
The results show that with 89% confidence level, our model is able to correctly predict the source of a news article between CNN and NYPost.
## 5.0	CONCLUSION
The news source prediction task is still relatively new and challenging, but thanks to innovative, state of the art systems like BERT, we were able to achieve impressive results. In this project, we attempted to build a model that predicts the source of a news article by taking advantage of the partisan bias (sentiment) of the news articles and evaluating the model. We began by crawling news sites for data, building our dataset, then processing and exploring our large dataset. After which we used the data to implement our model using BERT sentence embeddings. After evaluating the model it’s evident that our model performs well in this domain. There is still much work to be done in this area, we will continue to tune our current model, get more data, and incorporate more complex models.
## 5.1	Future Work
In future work, we would integrate news from a more “neutral” news source and make the problem a multiclass classification problem rather than just a binary classification problem. 
## References
https://news.gallup.com/poll/207794/six-partisan-bias-news-media.aspx

https://towardsdatascience.com/fse-2b1ffa791cf9

https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/
