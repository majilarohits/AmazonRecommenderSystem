import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# app = Flask(__name__, template_folder=os.path.abspath('.'), static_folder=os.path.abspath('static'))

app = Flask('__name__')

q = ""


@app.route("/")
def homePage():
    return render_template('home.html', query='')

#
@app.route("/predict", methods=['POST'])
def predict():
    # Load the training data
    df = pd.read_csv('amazon_co-ecommerce.csv')


    df['product_reviews'] = df['customer_reviews'].str.split('//')
    mixed_reviews_df = df.customer_reviews.apply(lambda x: pd.Series(str(x).split("//")))
    mixed_reviews_df.replace(np.nan, " ")
    newdf1 = mixed_reviews_df.loc[:, [0, 1]]
    newdf2 = mixed_reviews_df.loc[:, [4, 5]]
    newdf2 = newdf2.rename(columns={4: 0, 5: 1})
    newdf3 = mixed_reviews_df.loc[:, [8, 9]]
    newdf3 = newdf3.rename(columns={8: 0, 9: 1})
    newdf1 = newdf1.reset_index(drop=True)
    newdf2 = newdf2.reset_index(drop=True)
    newdf3 = newdf3.reset_index(drop=True)

    newdf = pd.concat([newdf1, newdf2, newdf3], axis=0, ignore_index=True)
    newdf = newdf.rename(columns={0: 'reviewText', 1: 'overall'})
    newdf['overall'] = newdf['overall'].astype(str)

    newdf['overall'] = newdf['overall'].str.extract(r'(\d+\.\d+)', expand=False)
    newdf['overall'] = newdf['overall'].str.replace(',', '')
    newdf['overall'] = pd.to_numeric(newdf['overall'])
    newdf['reviewText'] = newdf['reviewText'].astype(str)
    newdf.dropna()

    import random

    class Sentiment:
        NEGATIVE = "NEGATIVE"
        POSITIVE = "POSITIVE"

    class Review:
        def __init__(self, text, score):
            self.text = text
            self.score = score
            self.sentiment = self.get_sentiment()

        def get_sentiment(self):
            if self.score <= 3:
                return Sentiment.NEGATIVE
            else:
                return Sentiment.POSITIVE

    class ReviewContainer:
        def __init__(self, reviews):
            self.reviews = reviews

        def get_text(self):
            return [x.text for x in self.reviews]

        def get_sentiment(self):
            return [x.sentiment for x in self.reviews]

        def evenly_distribute(self):
            negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
            positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))

            print(len(negative))
            print(len(positive))
            print(negative[0].text)

            positive_shrunk = positive[:len(negative)]
            self.reviews = negative + positive_shrunk
            random.shuffle(self.reviews)

    reviews = []
    for i in range(len(newdf)):
        review = {
                'reviewText': newdf.loc[i, 'reviewText'],
                'overall': newdf.loc[i, 'overall']
        }
        reviews.append(Review(review['reviewText'], review['overall']))

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(reviews, test_size=0.33, random_state=42)

    train_container = ReviewContainer(train)

    test_container = ReviewContainer(test)


    train_container.evenly_distribute()

    train_x = train_container.get_text()
    train_y = train_container.get_sentiment()

    test_container.evenly_distribute()

    test_x = test_container.get_text()
    test_y = test_container.get_sentiment()

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()

    train_x_vectors = vectorizer.fit_transform(train_x)
    test_x_vectors = vectorizer.transform(test_x)

    from sklearn.linear_model import LogisticRegression

    clf_log = LogisticRegression()

    clf_log.fit(train_x_vectors, train_y)

    clf_log.predict(test_x_vectors[0])





    inputQuery1 = request.form['query1']





    test_set = [inputQuery1]

    # transform the test_set using our vectorizer
    new_test = vectorizer.transform(test_set)

    modelresult = clf_log.predict(new_test.toarray())

    if modelresult == 'POSITIVE':
        o1 = "Positive Review"
    if modelresult == 'NEGATIVE':
        o1 = "Negative Review"

    return render_template('home.html', output1=o1, query1=request.form['query1'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
