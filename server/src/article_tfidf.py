from utils import APP_PATH, SERVER_PATH, DATA_PATH, MODELS_PATH
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import argsort,array, dot, transpose
from pandas import read_csv, isnull
import os
import pickle

articles_path = os.path.join(DATA_PATH, "articles_3.csv")

dataframe_path = os.path.join(MODELS_PATH, "dataframe.pkl")
tfidf_vectorizer_path = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")
document_term_matrix_path = os.path.join(MODELS_PATH, "document_term_matrix.pkl")

class TfidfModels:
	def __init__(self):
		return

	def save_models(self):
		pickle.dump(self.dataframe, open(dataframe_path, "wb"))
		pickle.dump(self.tfidf_vectorizer, open(tfidf_vectorizer_path, "wb"))
		pickle.dump(self.document_term_matrix, open(document_term_matrix_path, "wb"))
		return

	def load_models(self):
		try:
			dataframe = pickle.load(open(dataframe_path, "rb"))
			tfidf_vectorizer = pickle.load(open(tfidf_vectorizer_path, "rb"))
			document_term_matrix = pickle.load(open(document_term_matrix_path, "rb"))
			return dataframe, tfidf_vectorizer, document_term_matrix
		except:
			print("No existing models to load.")
			return None, None, None

	def set_models(self, dataframe, tfidf_vectorizer, document_term_matrix):
		if dataframe is not None or tfidf_vectorizer is not None or document_term_matrix is not None:
			self.dataframe = dataframe
			self.tfidf_vectorizer = tfidf_vectorizer
			self.document_term_matrix = document_term_matrix
		return

	def load_set_models(self):
		dataframe, tfidf_vectorizer, document_term_matrix = self.load_models()
		self.set_models(dataframe, tfidf_vectorizer, document_term_matrix)
		return

class ArticleFetcher():
	def __init__(self, tfidf_model):
		self.tfidf_model = tfidf_model
		return

	def search_top_k(self, query, top_k=5):
		"""
		Search for the top k documents using TF-IDF

		Args:
				query (str): The query to look for
				top_k (int, optional): Top k documents that should be returned. Defaults to 5.

		Returns:
				[list]: Article IDs sorted by most similar to least similar
		"""
		transform_query = self.tfidf_model.tfidf_vectorizer.transform([query]) # Transform the query to maintain consistency
		similarities = dot(transform_query, transpose(self.tfidf_model.document_term_matrix))
		x = array(similarities.toarray()[0]) # temp of similarities as np array
		article_ids = argsort(x)[-top_k:][::-1] # most similar at the end -> get the last top_k using [-top_k] -> reverse the order using [::-1]
		article_details = self.get_article_details(article_ids)
		return article_details

	def get_article_details(self, article_ids):
		relevant_article_details = []
		for article_id in article_ids:
			df_row = self.tfidf_model.dataframe.iloc[article_id]
			response_object = {
				"id": int(article_id),
				"title": df_row["title"],
				"publication": df_row["publication"],
				"author": df_row["author"] if not isnull(df_row["author"]) else None,
				"date": df_row["date"],
				"content": df_row["content"]
				}
			relevant_article_details.append(response_object)
		return relevant_article_details

def create_document_term_matrix(tfidf_vectorizer, dataframe):
		article_content = dataframe["content"]
		document_term_matrix = tfidf_vectorizer.fit_transform(article_content)
		return document_term_matrix

def start():
	"""
	Used to recreate the models if needed (eg. using a different dataset, different stopwords, different parameters, etc)
	"""

	# Fresh start, read the articles, create the document term matrix, and dump the models for future use
	# Can call start() in if__name__ == "__main__"
	from nltk.corpus import stopwords
	df = read_csv(articles_path, encoding="utf-8", index_col=1)
	tfidf_vectorizer = TfidfVectorizer(
		max_df=0.25, # Drop drop that occur more than x% of the time
		min_df=5, # Only use words that appear at least x times
		stop_words=stopwords.words("english"), # Use the NLTK stopwords
		lowercase=True,
		use_idf=True,
		smooth_idf=True
	)
	df.drop(labels=["url", "year", "month", "Unnamed: 0"], inplace=True, axis=1)
	df.info()
	document_term_matrix = create_document_term_matrix(tfidf_vectorizer, dataframe)
	tfidf_models = TfidfModels()
	tfidf_models.set_models(df, tfidf_vectorizer, document_term_matrix)
	tfidf_models.save_models()
	return

if __name__ == "__main__":
	start()
