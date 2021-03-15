from flask import Flask, render_template, request, redirect, url_for, Response

import re
import math
import pickle
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)


@app.route('/')
def login():
    return render_template('index.html')


def clean(post):
    lbl_rmv = ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']

    stopw = {'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any',
             'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
             'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during',
             'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having',
             'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn',
             "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't",
             'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
             'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan'"shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't",
             'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these',
             'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't",
             'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
             'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
             'yourself', 'yourselves'}

    post = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', post)
    posts = re.sub("[^a-zA-Z]", " ", post)
    post = re.sub(' +', ' ', post).lower()
    # print("2="+post)
    for j in range(0, 16):
        post = re.sub(lbl_rmv[j], ' ', post)

    post = post.strip()
    # print("3="+post)

    post = re.sub('\s+', ' ', post)
    post = post.lower()
    post = post.split()
    # print("1="+post)
    post = [word for word in post if not word in stopw]
    ps = PorterStemmer()
    post = [ps.stem(word) for word in post]
    post = ' '.join(post)

    return post

def percent(pt):
    se = int(round(pt[1] * 100))
    if se == 50:
        if pt[1]>pt[0]:
            se+=1
        else:
            se-=1
    l = [str(se)+"%", str(100-se)+"%"]
    return l


def predict_data(data):

    ret = []

    map1 = ["I", "E"]
    map2 = ["N", "S"]
    map3 = ["T", "F"]
    map4 = ["J", "P"]

    data = clean(data)
    filename = "static/models/countvet.pickel"
    cv = pickle.load(open(filename, 'rb'))
    post = cv.transform([data]).toarray()

    iefile = "static/models/iemodel.sav"
    IEB = pickle.load(open(iefile, 'rb'))

    nsfile = "static/models/nsmodel.sav"
    NSB = pickle.load(open(nsfile, 'rb'))

    tffile = "static/models/tfmodel.sav"
    TFB = pickle.load(open(tffile, 'rb'))

    jpfile = "static/models/jpmodel.sav"
    JPB = pickle.load(open(jpfile, 'rb'))

    a = map1[IEB.predict(post)[0]]
    ret = percent(IEB.predict_proba(post)[0])

    b = map2[NSB.predict(post)[0]]
    ret += percent(NSB.predict_proba(post)[0])

    c = map3[TFB.predict(post)[0]]
    ret += percent(TFB.predict_proba(post)[0])

    d = map4[JPB.predict(post)[0]]
    print("JPB prob : ", end="")
    # print(JPB.predict_proba(post))
    ret += percent(JPB.predict_proba(post)[0])

    rett = a + b + c + d
    ret+=["Prediction : "+rett]

    return ret


@app.route('/result', methods=['GET', 'POST'])
def employee():
    if request.method == 'POST':
        result = request.form
        data = result["input"]
        print(data)
        ret = False
        ret = predict_data(data)
        if ret:
            return render_template('res.html', name=ret)
        else:
            return "Something went wrong !!"


if __name__ == '__main__':
    app.run(debug=True)
