from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv("spam_clean.csv")
    df.dropna(inplace=True)
    from sklearn.model_selection import train_test_split


    X = df['Text_Cleaned']
    y = df['Output']
    cv = TfidfVectorizer()
    X = cv.fit_transform(X) # Fit the Data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    stopwords = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', \
             'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he', 'her', 'hers', 'his', \
             'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or', \
             'see', 'seen', 'she', 'so', 'than', 'that', 'the', 'their', 'there', 'they', 'this', \
             'to', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'you']
# RUN THIS CELL TO ADD STOPWORDS TO THE LINEAR SVC PIPELINE:
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
