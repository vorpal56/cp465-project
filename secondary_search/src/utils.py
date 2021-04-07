from flask import make_response
import os
import pickle

APP_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SERVER_PATH = os.path.join(APP_PATH, "server")
DATA_PATH = os.path.join(SERVER_PATH, "data")
MODELS_PATH = os.path.join(DATA_PATH, "models")

def create_response(status_code, body):
	'''
	Function that creates a Flask Response with proper headers
	Params:
		status_code (int)
		body (str)
	Returns:
		response (flask.Response): Flask Response object with status code and body
	'''
	response = make_response()
	response.status_code = status_code
	response.data = body
	response.headers['Access-Control-Allow-Origin']='*'
	return response

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
