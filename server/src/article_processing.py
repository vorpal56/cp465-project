import os
import pickle
import nltk
from utils import APP_PATH, SERVER_PATH, DATA_PATH, MODELS_PATH, NUM_CLUSTERS, preprocess_string, clean_punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from numpy import argsort,array, dot, transpose, where
from pandas import read_csv, isnull
from itertools import permutations

ARTICLES_PATH = os.path.join(DATA_PATH, "articles_full.csv")

DATAFRAME_PATH = os.path.join(MODELS_PATH, "dataframe.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")
DOCUMENT_TERM_MATRIX_PATH = os.path.join(MODELS_PATH, "document_term_matrix.pkl")
KMEANS_MODEL_PATH = os.path.join(MODELS_PATH, "kmeans_model.pkl")

class RetrievalModels:
	MAX_N_CLUSTER_TERMS = 20 # The higher the number, the higher the probability that the same terms will appear in multiple clusters
	def __init__(self):
		return

	def save_models(self):
		pickle.dump(self.dataframe, open(DATAFRAME_PATH, "wb"))
		pickle.dump(self.tfidf_vectorizer, open(TFIDF_VECTORIZER_PATH, "wb"))
		pickle.dump(self.document_term_matrix, open(DOCUMENT_TERM_MATRIX_PATH, "wb"))
		pickle.dump(self.kmeans_model, open(KMEANS_MODEL_PATH, "wb"))
		return

	def load_models(self):
		try:
			dataframe = pickle.load(open(DATAFRAME_PATH, "rb"))
			tfidf_vectorizer = pickle.load(open(TFIDF_VECTORIZER_PATH, "rb"))
			document_term_matrix = pickle.load(open(DOCUMENT_TERM_MATRIX_PATH, "rb"))
			kmeans_model = pickle.load(open(KMEANS_MODEL_PATH, "rb"))
			return dataframe, tfidf_vectorizer, document_term_matrix, kmeans_model
		except:
			print("No existing models to load. Run the article_processing.py script")
			return None, None, None, None

	def set_models(self, dataframe, tfidf_vectorizer, document_term_matrix, kmeans_model):
		self.dataframe = dataframe
		self.tfidf_vectorizer = tfidf_vectorizer
		self.document_term_matrix = document_term_matrix
		self.kmeans_model = kmeans_model
		return

	def load_set_models(self):
		dataframe, tfidf_vectorizer, document_term_matrix, kmeans_model = self.load_models()
		self.set_models(dataframe, tfidf_vectorizer, document_term_matrix, kmeans_model)
		return

class ArticleFetcher():
	def __init__(self, retrieval_models):
		self.retrieval_models = retrieval_models
		word_lists, self.secondary_index = self.get_top_n_cluster_terms()
		cluster_terms = {i: words for i, words in enumerate(word_lists)}
		return

	def search_top_k_tfidf(self, query, top_k=5, tfidf_vectorizer=None, document_term_matrix=None, subdf=None):
		"""
		Search for the top k documents using TF-IDF using either a subset of articles or the entire corpus of articles

		Args:
				query (str): The query string searched
				top_k (int, optional): Top k documents that should be returned. Defaults to 5.
				tfidf_vectorizer (TfidfVectorizer, optional): A TfidfVectorizer with different params than retrieval_models. Defaults to None.
				document_term_matrix (matrix, optional): A document term matrix with different weights than retrieval_models. Defaults to None.
				subdf (DataFrame, optional): The subset dataframe if any. Defaults to None.
		Returns:
				list(dict): Article details sorted by most similar to least similar.
		"""
		query = clean_punctuation(preprocess_string(query))
		if tfidf_vectorizer is not None:
			transform_query = tfidf_vectorizer.transform([query])
		else:
			transform_query = self.retrieval_models.tfidf_vectorizer.transform([query]) # Transform the query to maintain consistency (eg. use nltk stopwords, lowercase, etc.)
		if document_term_matrix is not None:
			similarities = dot(transform_query, transpose(document_term_matrix))
		else:
			similarities = dot(transform_query, transpose(self.retrieval_models.document_term_matrix))
		x = array(similarities.toarray()[0]) # secondary_index of similarities as np array
		article_ids = argsort(x)[-top_k:][::-1] # most similar at the end -> get the last top_k using [-top_k] -> reverse the order using [::-1]
		article_details = self.get_article_details(article_ids, subdf=subdf)
		return article_details

	def search_top_k_secondary_index(self, query, top_k=5):
		"""Search for the top k documents using the secondary index (which points to a list of cluster indexes which then point to a subset of articles) and TF-IDF on the subset of articles

		Args:
				query (str): The query string searched
				top_k (int, optional): Top k documents that should be returned. Defaults to 5.

		Returns:
				list(dict): Article details sorted by most similar to least similar.
		"""
		query = clean_punctuation(preprocess_string(query))
		cluster_indexes = self.secondary_index.get(query)
		transform_query = self.retrieval_models.tfidf_vectorizer.transform([query])
		# print(transform_query)
		if cluster_indexes is None:
			cluster_indexes = self.retrieval_models.kmeans_model.predict(transform_query).tolist()
		if len(cluster_indexes) == 1:
			article_ids = where(self.retrieval_models.kmeans_model.labels_ == cluster_indexes[0])[0].tolist()
		else:
			article_ids = set() # the order of documents is not significant in sets which is why the order can be a bit different. the order for the purpose of evaluating similarities is not relevant
			for cluster_index in cluster_indexes:
				cluster_article_ids = where(self.retrieval_models.kmeans_model.labels_ == cluster_index)[0].tolist()
				article_ids.update(cluster_article_ids)
		# problem is the articles are ordered based on when they're inserted, so we'll use TF-IDF again, but with subset of articles that are clustered based on the labels
		subdf = self.retrieval_models.dataframe.iloc[list(article_ids)]
		tfidf_vectorizer = TfidfVectorizer(
			stop_words=nltk.corpus.stopwords.words("english"), # Use the NLTKs stopwords
			lowercase=True, use_idf=True, smooth_idf=True
		) # Use a generic TF-IDF vectorizer which doesn't have any limiters
		document_term_matrix = create_document_term_matrix(tfidf_vectorizer, subdf["cleaned"])
		# search using the same search_top_k_tfidf function
		article_details = self.search_top_k_tfidf(query, top_k, tfidf_vectorizer, document_term_matrix, subdf)
		return article_details

	def get_article_details(self, article_ids, subdf=None):
		"""Creates a list of article objects with relevant details

		Args:
				article_ids (list|set): Article IDs to get details about (sorted by most similar to least similar).
				subdf (DataFrame, optional): The subset dataframe if any. Defaults to None.

		Returns:
				list(dict): Article details sorted by most similar to least similar.
		"""
		relevant_article_details = []
		for article_id in article_ids:
			df_row = self.retrieval_models.dataframe.iloc[article_id] if subdf is None else subdf.iloc[article_id]
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

	def get_top_n_cluster_terms(self, n=10):
		n = max(n, RetrievalModels.MAX_N_CLUSTER_TERMS)
		terms = self.retrieval_models.tfidf_vectorizer.get_feature_names()
		centers = self.retrieval_models.kmeans_model.cluster_centers_.argsort()[:, ::-1]
		word_lists = [[terms[j] for j in centers[i, :n]] for i in range(NUM_CLUSTERS)] # get the closest n terms to the center of each cluster
		secondary_index = {}
		# The idea is that the "secondary index" has the words which points to a cluster index which points to a subset of articles
		for cluster_index, word_list in enumerate(word_lists):
			for word_position, word in enumerate(word_list):
				if word in secondary_index:
					secondary_index[word].append(cluster_index)
				else:
					secondary_index[word] = [cluster_index]
		return word_lists, secondary_index

def create_document_term_matrix(tfidf_vectorizer, articles_info):
	document_term_matrix = tfidf_vectorizer.fit_transform(articles_info)
	return document_term_matrix

def create_df(path=None):
	articles_path = ARTICLES_PATH if path is None else path
	df = read_csv(articles_path, encoding="utf-8").reset_index()
	# df.astype('str')
	df = df.filter(items=["id", "title", "publication", "author", "date", "content"])
	# df.drop(labels=["url", "year", "month", "Unnamed: 0"], inplace=True, axis=1)
	return df

def start():
	"""
	Used to recreate the models if needed (eg. using a different dataset, different stopwords, different parameters, etc)
	"""

	# Fresh start, read the articles, create the document term matrix, and dump the models for future use
	# Can call start() in if__name__ == "__main__"
	nltk.download("stopwords")
	tfidf_vectorizer = TfidfVectorizer(
		max_df=0.25, # Drop words that occur in more than x% of documents -> kind of like stopwords
		min_df=5, # Drop words that occur in less than x documents -> really rare words along a large corpus doesn't help. Try to use a number proportional to the number of documents used for the models
		stop_words=nltk.corpus.stopwords.words("english"), # Use the NLTKs stopwords
		lowercase=True,
		use_idf=True,
		smooth_idf=True,
		# ngram_range=(1, 2) # (x, y): vectorization can consist of x to y words (eg. software engineering = [software, engineering] or software engineering)
	)
	df = create_df()
	print("Preprocessing article features...")
	df["content"] = df["content"].apply(preprocess_string)
	df["cleaned"] = df["content"].apply(clean_punctuation) # Save computation time by creating a new column with the cleaned data
	print("Creating document term matrix...")
	document_term_matrix = create_document_term_matrix(tfidf_vectorizer, df["cleaned"])
	print("Fitting document term matrix to KMeans...")
	kmeans_model = MiniBatchKMeans(n_clusters=NUM_CLUSTERS)
	kmeans_model.fit(document_term_matrix)
	retrieval_models = RetrievalModels()
	print("Setting and saving models...")
	retrieval_models.set_models(df, tfidf_vectorizer, document_term_matrix, kmeans_model)
	retrieval_models.save_models()
	print("Finished")
	return

if __name__ == "__main__":
	# ARTICLES_PATH = os.path.join(DATA_PATH, "articles_3.csv")
	# a = create_df(path=ARTICLES_PATH)
	# # a.info()
	# k = a.iloc[[2000, 3000, 4000, 5000]]
	# print(k.head())
	# -----------------separate---------------
	start()
