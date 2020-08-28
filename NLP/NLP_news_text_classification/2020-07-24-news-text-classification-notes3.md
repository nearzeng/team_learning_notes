---
title: News Text Classification with ML
date: 2020-07-24 22:09:09
tags:
- NLP

categories:
- 入门
- NLP
- classification

description: apply some ML algorithm in News Text Classification.

photos:
- http://picture.ik123.com/uploads/allimg/171116/4-1G116105229.jpg
---

## 1. Count Vectors + RidgeClassifier

```python
# Count Vectors + RidgeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('/kaggle/input/newsclassestestdata/train_set.csv/train_set.csv', sep='\t', nrows=15000)

vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
```

**output**: 0.7410794074418383

## 2. TF-IDF + RidgeClassifier

```python
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv('/kaggle/input/newsclassestestdata/train_set.csv/train_set.csv', sep='\t', nrows=15000)
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
```

**output**: 0.8721598830546126

Try a bigger max_features:

```python
tfid_try = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
train_try = tfid_try.fit_transform(train_df['text'])
clf_try = RidgeClassifier()
clf_try.fit(train_try[:10000], train_df['label'].values[:10000])
val_pred_try = clf_try.predict(train_try[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred_try, average='macro'))
```

**output**: 0.8850817067811825

## 3. LogisticRegression

```python
from sklearn import linear_model

tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=5000)
train_test = tfidf.fit_transform(train_df['text'])

reg = linear_model.LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
reg.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = reg.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
```

**output**: 0.8464704900433653

## 4. SGDClassifier

```python
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=5000)
train_test = tfidf.fit_transform(train_df['text'])

reg = linear_model.SGDClassifier(loss="log", penalty='l2', alpha=0.0001,l1_ratio=0.15) 
reg.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = reg.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
```

**output**: 0.8461511856339045

## 5. SVM

```python
from sklearn import svm
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=5000)
train_test = tfidf.fit_transform(train_df['text'])

reg = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',decision_function_shape='ovr')
reg.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = reg.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
```

**output**: 0.883129115819089

## 6. Summary

| method                          | f1_score               |
| ------------------------------- | ---------------------- |
| Count Vectors + RidgeClassifier | 0.7410794074418383     |
| TF-IDF + RidgeClassifier        | **0.8850817067811825** |
| TF-IDF + LogisticRegression     | 0.8464704900433653     |
| TF-IDF + SGDClassifier          | 0.8461511856339045     |
| TF-IDF + SVM                    | **0.883129115819089**  |

