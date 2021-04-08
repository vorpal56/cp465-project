from string import punctuation
from functools import wraps
import os
import pickle
import re
import time
import json

APP_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SERVER_PATH = os.path.join(APP_PATH, "server")
DATA_PATH = os.path.join(SERVER_PATH, "data")
MODELS_PATH = os.path.join(DATA_PATH, "models")

NUM_CLUSTERS = 15 # The number of clusters we want to use -> depends on how large the dataset is

punctuation = punctuation + '\n' + '—“,”‘-’' + '0123456789' # punctuation specifically for the article string

def preprocess_string(string, remove_punc=False):
	string = re.sub('(\S+@\S+)(com|\s+com)', ' ', string) # email address/usernames
	if remove_punc:
		string = clean_punctuation(string)
	string = re.sub('\s{1,}', ' ', string) # extra whitespace
	string = " ".join([word for word in string.split() if len(word) > 2])
	return string

def clean_punctuation(string):
	return ''.join(word for word in string if word not in punctuation)

def time_call(calling_function):
	@wraps(calling_function)
	def wrapper(*args, **kwargs):
		start_time = time.time()
		results = calling_function(*args, **kwargs)
		end_time = time.time() - start_time
		response_obj = create_response(results, end_time)
		return response_obj
	return wrapper

def create_response(article_details, end_time):
	obj = {
		"time": end_time,
		"articles": article_details
	}
	return json.dumps(obj)

def compile_sample():
	import pandas as pd
	import random
	# sample 10% of the data (about 5000 articles)
	portion = 0.1
	df = pd.read_csv(
		os.path.join(DATA_PATH, "articles_full.csv"),
		encoding="utf-8",
		header=0,
		skiprows=lambda index: index > 0 and random.random() > portion
	)
	df.to_csv(os.path.join(DATA_PATH, "articles_3.csv"), index=False, encoding="utf-8")
	print("ok")
	return

if __name__ == "__main__":
	compile_sample()
