# Standdown Exercise

The cell below stores the text of a set of famous books in the variable nltk_books.


```python
# Run cell with no changes

import nltk
import pandas as pd

# store raw text of books in a list
nltk_books = [nltk.corpus.gutenberg.raw(title) 
                 for title in nltk.corpus.gutenberg.fileids()]

# convert list to dataframe with titles as the index.
nltk_books = pd.DataFrame(all_sentences, 
                          index=nltk.corpus.gutenberg.fileids(),
                          columns=['raw_text'] )
```

The next cell below splits the books into a train and test sets.  This is an arbitrary split, but is here to remind you that we fit a vectorizer only on the training set.


```python
# Run cell with no changes
from sklearn.model_selection import train_test_split

train, test = train_test_split(nltk_books, random_state=42)

```


```python
# Here are the books whose full texts compose the training set
train.index
```




    Index(['milton-paradise.txt', 'shakespeare-macbeth.txt',
           'shakespeare-hamlet.txt', 'edgeworth-parents.txt', 'austen-sense.txt',
           'chesterton-brown.txt', 'whitman-leaves.txt', 'blake-poems.txt',
           'melville-moby_dick.txt', 'carroll-alice.txt',
           'chesterton-thursday.txt', 'shakespeare-caesar.txt',
           'burgess-busterbrown.txt'],
          dtype='object')



Your task is to fit a TfidfVectorizer to the training set with the following specifications: max_features is set to 50, stopwords are removed using the nltk english stopwords list.  The other parameters should be the defaults.  

After fitting the vectorizer, find the word with the highest tf-idf score in Moby Dick.    

> Hint: Converting the vectorized text into a DataFrame with column names and indices will make your life easier.  Use the following hints to make that happen:  
>> 1. The TF-IDF vectorizer returns a sparse matrix.  Chain the toarray() method off the vectorizer, then convert that array into a DataFrame.  

>> 2. The fit Tf-Idf object has a method called `get_feature_names()`. Assign the result of that method as the `columns` argument of DataFrame.  

>> 3. Pass train.index as the index argument of DataFrame.   
    




```python

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

tfidf = TfidfVectorizer(max_features=50, stop_words=stopwords.words('english'))

train_vec = pd.DataFrame(tfidf.fit_transform(train['raw_text']).toarray(),
             columns=tfidf.get_feature_names(), 
            index=train.index)
train_vec.loc['melville-moby_dick.txt'].sort_values(ascending=False)
```




    whale      0.767525
    one        0.255945
    upon       0.210215
    like       0.179800
    man        0.146453
    sea        0.145935
    old        0.125054
    would      0.120052
    though     0.106713
    thou       0.100650
    head       0.095875
    yet        0.095875
    time       0.092818
    long       0.092540
    still      0.086704
    great      0.085037
    said       0.084481
    two        0.082814
    last       0.082683
    every      0.080021
    must       0.078645
    us         0.078641
    see        0.075588
    way        0.075311
    never      0.071053
    first      0.070146
    little     0.069197
    men        0.067807
    say        0.067807
    may        0.066696
    much       0.066564
    well       0.063917
    good       0.060026
    could      0.060026
    go         0.053912
    thee       0.052490
    thing      0.052245
    might      0.050855
    come       0.049744
    made       0.049466
    day        0.048910
    let        0.043908
    know       0.042241
    thought    0.041685
    thy        0.038976
    think      0.033904
    make       0.031125
    mr         0.029643
    shall      0.028954
    mrs        0.006674
    Name: melville-moby_dick.txt, dtype: float64


