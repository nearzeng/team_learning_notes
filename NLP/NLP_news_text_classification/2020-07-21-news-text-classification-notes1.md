---
title: News Text Classification —— Tianchi Competition Understanding
date: 2020-07-21 08:42:45
tags:
- NLP

categories:
- 入门
- NLP
- classification

description: intros and understanding about a competition —— https://tianchi.aliyun.com/competition/entrance/531810/introduction

photos:
- http://picture.ik123.com/uploads/allimg/171116/4-1G116105224.jpg
---

## 1. Intros

I joined in a NLP team learning activity held by an open-source organization [Datawhale](https://datawhale.club/) again. This time, we're gonna explore News Text Classification Challenge.

This is a classic text classification problem which requires players to classify news categories based on news text characters. And this challenge is held by Datawhale and Tianchi, and with the help of Datawhale, we can learn more about the knowledge points of NLP preprocessing, model construction and model training.

Here's the competition: https://tianchi.aliyun.com/competition/entrance/531810/introduction

## 2. Understanding

### 2.1 Data

The contest question use anonymously processed news data as contest question data, and the dataset is visible and downloadable after registration. The contest question data is news text and is anonymized according to character level. Integrate and divide 14 candidate classification categories: finance, lottery, real estate, stocks, home furnishing, education, technology, society, fashion, current affairs, sports, constellation, games and entertainment text data.

The question data consists of the following parts: 20w samples in the training set, about 5w samples in the test set A, and about 5w samples in the test set B. 

The source of the question data is news on the Internet, which is collected and processed anonymously. Therefore, the contestants can perform data analysis on their own, and can give full play to their strengths to complete various feature projects, without restricting the use of any external data and models.

### 2.2 Label

The corresponding relationship of the labels in the data set is as follows: 

> {'Technology': 0,'Stocks': 1,'Sports': 2,'Entertainment': 3,'Current Affairs': 4,'Society': 5,'Education' : 6,'Finance': 7,'Home Furnishing': 8,'Game': 9,'Property': 10,'Fashion': 11,'Lottery': 12,'Constellation': 13}

training set columns:

| label | text                                                         |
| ----- | ------------------------------------------------------------ |
| 2     | 2967 6758 339 2021 1854 3731 4109 3792 4149 1519 2058 3912 2465 2410 1219 6654 7539 264 2456 4811 12... |

### 2.3 Evaluation

The evaluation standard is the average value of the category `f1_score`. The results submitted by the players are compared with the categories of the actual test set. The larger the result, the better.

formula:
$$
F1 = \frac{1}{\frac{1}{2} * (\frac{1}{precision} + \frac{1}{recall})}= 2 * \frac{(precision * recall)}{(precison + recall)}
$$
Mathematically, F1-Socre is defined as the harmonic mean of precision and recall. From the formula, we can see that the size of F1 is affected by Precision and Recall, that is, the short board effect, so F1 Score is more balanced than the direct average result, and it can better explain the quality of a model.

coding example:

```python
from sklearn.metrics import f1_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
f1_score(y_true, y_pred, average='macro')
```

output: 0.26666666666666666



### 2.4 Solving Ideas

**Analysis**: The essence of this challenge is a text classification problem, which needs to be classified according to the characters of each sentence. However, the data given in the question is anonymized, and operations such as Chinese word segmentation cannot be used directly. This is the difficulty of this challenge.

Therefore, the difficulty of this competition is the need to model anonymous characters to complete the process of text classification. Since text data is a typical unstructured data, it may involve two parts: `feature extraction` and `classification model`. In order to reduce the difficulty of the competition, Datawhale has provided some ideas for solving problems for your reference:

##### Idea 1: TF-IDF + machine learning classifier

Use TF-IDF to extract features directly from the text, and use the classifier to classify. In the choice of classifier, SVM, LR, or XGBoost can be used.

##### Idea 2: FastText

FastText is an entry-level word vector. Using the FastText tool provided by Facebook, you can quickly build a classifier.

##### Idea 3: WordVec + Deep Learning Classifier

WordVec is an advanced word vector, and the classification is completed by constructing a deep learning classification. The network structure of deep learning classification can choose TextCNN, TextRNN or BiLSTM.

##### Idea 4: Bert word vector

Bert is a highly matched word vector with powerful modeling and learning capabilities.

## 3. What's Next?

Read the data in and try idea 1.

## 4. References

##### 1. [Task1 赛题理解](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.6.6406111aIKCSLV&postId=118252) 

##### 2. [通过实例来梳理概念 ：准确率 (Accuracy)、精准率(Precision)、召回率(Recall)和 F值](https://mp.weixin.qq.com/s/d5d3b3tnetqYKuckfUI08w) 