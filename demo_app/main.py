from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob
from fastapi.middleware.cors import CORSMiddleware

job_postings = pd.read_csv('job_postings.csv')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize text into individual words
    tokens = word_tokenize(text.lower())
    # Remove stop words from tokens
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize filtered tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Return cleaned text as a string
    return ' '.join(lemmatized_tokens)

job_postings['cleaned_text'] = job_postings['text'].astype(str).apply(clean_text)


inclusive_keywords = ['equal opportunity', 'diversity', 'inclusion', 'minority', 'gender', 'race', 'ethnicity']

criteria = {'Gender': ['woman', 'man', 'non-binary', 'transgender', 'genderqueer'],
            'Race/Ethnicity': ['Black', 'Hispanic/Latinx', 'Asian', 'White', 'Native American', 'Middle Eastern'],
            'Age': ['young', 'old', 'experienced', 'fresh'],
            'Disability': ['disabled', 'handicapped', 'wheelchair user', 'able-bodied'],
            'LGBTQ+': ['lesbian', 'gay', 'bisexual', 'queer', 'trans', 'non-heterosexual', 'non-cisgender']}

scoring = {'Gender': 4,
           'Race/Ethnicity': 3,
           'Age': 1,
           'Disability': 1,
           'LGBTQ+': 2}

def get_inclusive_score(description):
    # Clean the job description text
    cleaned_text = clean_text(description)
    # Split cleaned text into individual words
    words = cleaned_text.split()
    # Calculate total number of words in job posting
    total_words = len(words)
    # Calculate number of occurrences of each inclusive keyword
    keyword_counts = [words.count(keyword.lower()) for keyword in inclusive_keywords]
    # Calculate total number of inclusive keywords
    total_keywords = sum(keyword_counts)
    # Calculate inclusive score as a percentage of total words
    inclusive_score = total_keywords / total_words * 100
    # Return the inclusive score
    return inclusive_score

def get_sentiment_score(description):
    # Clean job description text
    cleaned_text = clean_text(description)
    # Get TextBlob object for cleaned text
    blob = TextBlob(cleaned_text)
    # Calculate sentiment score using TextBlob
    sentiment_score = blob.sentiment.polarity
    return sentiment_score


# Declaring our FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a route for the web application
@app.get("/predict")
def predict(review: str):
	inclusive_score = get_inclusive_score(review)
	sentiment_score = get_sentiment_score(review)
	Inclusive =("Inclusive Score: {:.2f}".format(inclusive_score))
	Sentiment =("Sentiment Score: {:.2f}".format(sentiment_score))
	if inclusive_score < 10:
    		return(str(Inclusive+'\n'+Sentiment+'\n'+"Suggestion: Consider adding more diversity words."))
	elif inclusive_score >= 10 and inclusive_score < 20:
    		return(str(Inclusive+'\n'+Sentiment+'\n'+"Suggestion: Consider adding words that promote inclusion based on sexual orientation."))
	else:
    		return(str(Inclusive+'\n'+Sentiment+'\n'+"This looks okay and you can post it now."))
	if sentiment_score == 0:
    		return(str(Inclusive+'\n'+Sentiment+'\n'+"Your Job description is neutral "))
	else:
    		return(str(Inclusive+'\n'+Sentiment+'\n'+"You may consider checking the language of your description"))
	

