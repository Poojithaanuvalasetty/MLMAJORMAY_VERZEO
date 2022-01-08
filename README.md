# ML MAJOR MAY

By [Poojitha Anuvalasetty](mailto:poojithaanuvala@gmail.com)



### Problem Statement

To design a ML model using SVM algorithm that helps predicting if a restaurant's food review left by a customer is either positive or negative.

### Description

We use sentiment analysis to train our model on words that tend to fall under the categories **positive** and **negative**. The given problem is to create a ML model which helps us in classifying these reviews. So, this is a classification problem.

### Dataset

The Dataset is taken from [kaggle](https://www.kaggle.com/d4rklucif3r/restaurant-reviews).

This Dataset contains two rows Customer Reviews and Liked.
Customer reviews tells us about the reviews given by the customers for a food in restaurant and liked column tells about whether they liked the food or not.

|   Customer Reviews    | Whether they liked food or not |
| :-------------------: | :----------------------------: |
| **997** unique values |      **705** total values      |

### Algorithm

Since this is a classification problem, we use SVM algorithm to build the ML model.

- SVM is supervised machine learning algorithm which can be used for either classification or regression problems. 
- Classification means to predict the label/group to which our prediction belongs to.
- SVM perform classification by finding the hyper-plane that differentiate the classes that we plotted in a n-dimensional space.
- SVM draws the hyperplane by transforming our data with the help of functions called **Kernels**.

### Steps 

Steps needed to build the model.

1. Import modules.
2. Loading the dataset.
3. Visualizing the data
4. Reviewing the data 
5. Vectorizing the text
6. Training and Splitting the dataset
7. Applying SVM and predicting our outputs
8. Accuracy
9. Saving the model 
10. User Check

### Making the Model

**1** Importing libraries and packages required for SVM

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
```

### **About the libraries:**

**NumPy:** NumPy is a Python library used for working with arrays

**Pandas:** Pandas is a Python library used for data analysis

**Matplotlib:** Matplotlib is a Python library used for data visualization and graphical plotting.

**NLTK:** Natural Language Toolkit is used for building python programs that work with human language data.



**2** Loading the dataset

```python
dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', encoding='utf8')
dataset.head()
```

Output:

```
	Review											Liked
0	Wow... Loved this place.							1
1	Crust is not good.									0
2	Not tasty and the texture was just nasty.			0
3	Stopped by during the late May bank holiday of...	1
4	The selection on the menu was great and so wer...	1
```



**3** Visualizing the data

```python
plt.rcParams["figure.figsize"] = (15,7)
dataset.plot.hist(dataset, color = "lightgreen")
```



![image-20220107233324211](C:\Users\akshi\AppData\Roaming\Typora\typora-user-images\image-20220107233324211.png)



**4** Reviewing the data 

Importing required packages from NLTK and reviewing the data

```python
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
ls= WordNetLemmatizer()
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review=review.lower()
    review=review.split()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    review=[ls.lemmatize(word) for word in review if not word in set(all_stopwords)]
    review=' '.join(review)
    corpus.append(review)
print(corpus)
```

Output:

```
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package wordnet to /root/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
['wow loved place', 'crust not good', 'not tasty texture nasty',....
```



**5** Vectorizing the text

Converting text into a vector based on frequency of occurrence of each word

```python
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values
len(x[0])
```

Output:

```
1767
```

Cutting down the maximum features

```python
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1566)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values
len(x[0])
```

Output:

```
1566
```



**6** Training and Splitting the dataset

```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=881)
```



**7** Applying SVM and predicting our outputs

```python
from sklearn.svm import SVC
clf=SVC(kernel='sigmoid',C=1)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_pred)
```

Output:

```
[1 0 1 0 1 0 0 1 0 0 1 1 1 0 0 1 1 0 0 0 0 1 1 0 1 0 1 1 0 1 1 0 0 0 1 0 0
 1 0 1 0 1 0 0 1 1 0 0 0 1 0 1 0 0 1 1 0 1 0 0 0 0 0 1 1 1 0 1 0 1 1 0 1 1
 1 0 0 0 1 0 0 0 1 0 1 0 1 1 0 0 0 0 1 0 1 1 0 0 1 1 1 0 0 0 1 0 0 1 0 0 1
 0 0 1 0 1 1 0 0 0 0 0 1 0 1 0 0 0 0 1 0 1 1 1 0 1 1 1 0 0 0 1 0 0 1 1 0 0
 1 1]
```



**8** Accuracy

Checking the accuracy of our model

```python
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)
```

Output:

```
0.9133333333333333
```



**9** Saving the Model

Using joblib to save the ML model and load it to test on unknown data

```python
import joblib
filename = 'finalizedmodel.sav'
joblib.dump(clf, filename)
```

Output:

```
['finalizedmodel.sav']
```

Loading the model to test on unknown data

```python
loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print(result)
```

Output:

```
0.9133333333333333
```



**10** User Check

Checking our model for predicting when the data input is taken from the user.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
ls= WordNetLemmatizer()
text=input("Enter your review:")
text=re.sub('[^a-zA-Z]',' ', text)
text=text.lower()
text=text.split()
ls=WordNetLemmatizer()
all_stopwords=stopwords.words('english')
all_stopwords.remove('not')
text=[ls.lemmatize(word) for word in text if not word in set(all_stopwords)]
text=' '.join(text)
print(text)
corpus.append(text)

def usercheck():
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer(max_features=1566)
    x=cv.fit_transform(corpus).toarray()
    pred=clf.predict([x[-1]])
    print(pred)
    if pred==1:
        print("Yay!! customer says the food was good!")
    else:
        print("Damn,customer says the food was bad!")

usercheck()
```

Output:

Positive

```
Enter your review:The food was delicious
food delicious
[1]
Yay!! customer says the food was good!
```

Negative

```
Enter your review:The food was disgusting
food disgusting
[0]
Damn,customer says the food was bad!
```



### Conclusion

- The accuracy obtained by the model is 91%
- The accuracy changes as we change the type of Kernel we use.
- The highest accuracy was obtained while using a **Sigmoid** Kernel, giving us 91.3% accuracy.
- **RBF** Kernel gave us an accuracy of 91%
- **Linear** Kernel also gave us an accuracy of 91%







