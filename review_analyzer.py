import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import plotly.graph_objs as go
import reviewscrapper

def preprocess_text(text):
    # Make text lowercase and remove links, text in square brackets, punctuation, and words containing numbers
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', ' ', text)
    text = re.sub(r'\n', ' ', text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words).strip()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lem_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    
    return ' '.join(lem_tokens)

def display_result(result):
    if result[0]=="Positive":
        st.subheader(result[0]+":smile:")
    elif result[0]=="Negative":
        st.subheader(result[0]+":pensive:")
    else:
        st.subheader(result[0]+":neutral_face:")

def classify_multiple(dataframe):
    st.write(f"There are a total of {dataframe.shape[0]} reviews given")

    dataframe.columns = ["Review"]
    data = dataframe.copy()
    data["Review"].apply(preprocess_text)
    count_positive = 0
    count_negative = 0
    count_neutral = 0
    sentiments = []
    for i in range(dataframe.shape[0]):
        rev = data.iloc[i]
        rev = vect.transform(rev)
        res = model.predict(rev)
        sentiments.append(res)
        if res[0]=='Positive':
            count_positive+=1
        elif res[0]=='Negative':
            count_negative+=1
        else:
            count_neutral+=1 

    x = ["Positive", "Negative", "Neutral"]
    y = [count_positive, count_negative, count_neutral]

    fig = go.Figure()
    layout = go.Layout(
        title = 'Product Reviews Analysis',
        xaxis = dict(title = 'Category'),
        yaxis = dict(title = 'Number of reviews'),)
    
    fig.update_layout(dict1 = layout, overwrite = True)
    fig.add_trace(go.Bar(name = 'Multi Reviews', x = x, y = y))
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"Positive: {count_positive}, Negative: {count_negative}, Neutral: {count_neutral}")
    dataframe["Sentiment"] = sentiments
    st.dataframe(dataframe)

if __name__ == "__main__":
    st.title('Sentiment Analysis of Amazon product reviews!')
    st.divider()
    classifier = st.radio(
        "Which classifier do you want to use?",
        ["Logistic Regression", "Support Vector Machine (SVM)"])
    if classifier == 'Logistic Regression':
        st.write('You selected Logistic Regression')
    else:
        st.write("You selected SVM")
    st.divider()
    
    with open("models.p", 'rb') as mod:
            data = pickle.load(mod)
    vect = data['vectorizer']

    if classifier=="Logistic Regression":
        model = data["logreg"]
    else:
        model = data["svm"]


    st.subheader('Check sentiments of a single review:')
    single_review = st.text_area("Enter review:")
    if st.button('Check the sentiment!'):
        review = preprocess_text(single_review)
        inp_test = vect.transform([single_review])
        result = model.predict(inp_test)
        print(result)
        display_result(result)
        
    else:
        st.write('')

    st.divider()
    st.subheader('Check sentiments of an Amazon product:')
    url_review = st.text_input("Enter the URL to the product:")
    if st.button('Check the reviews!'):
        reviewscrapper.web_scrapper(url_review)
        url_rev = pd.read_csv("scrapedReviews.csv")
        
        classify_multiple(pd.DataFrame(url_rev["Review"]))

        
    else:
        st.write('')
        
    st.divider()
    st.subheader('Check sentiments of multiple reviews:')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        if dataframe.shape[1]!=1:
            st.write("Wrong CSV format!")
        else:
            classify_multiple(dataframe)
        