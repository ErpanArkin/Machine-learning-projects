import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import CountVectorizer
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import confusion_matrix,classification_report
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
yelp = pd.read_csv('yelp_training_set_review(with text_length and transformed).csv')
yelp['text_transformed'] = yelp['text_transformed'].astype('str')
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]
X = yelp_class['text_transformed']
y = yelp_class['stars']
cv = CountVectorizer()
X_vector = cv.fit_transform(X)
training_features, testing_features, training_target, testing_target = \
            train_test_split(X_vector, y, random_state=None)

# Average CV score on the training set was: 0.9316421943875852
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    MultinomialNB(alpha=0.01, fit_prior=True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
print(classification_report(testing_target,results))
