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
    


