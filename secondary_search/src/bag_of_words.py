from nltk.corpus import stopwords
from nltk import  word_tokenize
from collections import Counter

def tokenize_article(article_content):
    words = []
    word_tokens = word_tokenize(article_content) #Breaks ALL words apart
    return word_tokens

def filter_stopwords(words):
    filtered_word_list = words[:] #make a copy of the word_list
    for word in words: # iterate over word_list
        if word in stopwords.words('english'):
            filtered_word_list.remove(word)

    for item in filtered_word_list:
        item.lower()
    return filtered_word_list
    
def bag_of_words_generator(article_content):   
    word_tokens = tokenize_article(article_content) 
    #Get a list of all words that are used
    filtered_words = filter_stopwords(word_tokens)
    #filtered_words now has our word list (with repeat) in a list without english stopwords
    occurances = Counter(filtered_words)
    #Counter returns a dictionary words are keys occurances are values
    #Measure of TERM FREQUENCY

    #For clustering, we need to feed the filtered_words array,
    #as K-means expects an array or a matrix

    return occurances
