from bag_of_words import bag_of_words_generator #Our word bag
import os
from utils import APP_PATH, SERVER_PATH, DATA_PATH, MODELS_PATH
import pickle


dataframe_path = os.path.join(MODELS_PATH, "dataframe.pkl")
dataframe = pickle.load(open(dataframe_path, "rb"))

word_bags = []

for article in dataframe["content"]:
    bag = bag_of_words_generator(article) #Make a word bag for an article
    word_bags.append(bag)


#Word bags have format [word1: occurances, word2: occurances], could not match id to content