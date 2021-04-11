import os
import pickle
import nltk
from utils import APP_PATH, SERVER_PATH, DATA_PATH, MODELS_PATH, NUM_CLUSTERS, preprocess_string, clean_punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from numpy import argsort,array, dot, transpose, where
from pandas import read_csv, isnull

ARTICLES_PATH = os.path.join(DATA_PATH, "articles_full.csv")
DATAFRAME_PATH = os.path.join(MODELS_PATH, "dataframe.pkl")
#------------------------------------------------------------------------------------------------
CONTENT_TFIDF_VECTORIZER_PATH = os.path.join(MODELS_PATH, "content_tfidf_vectorizer.pkl")
GENERIC_TFIDF_VECTORIZER_PATH = os.path.join(MODELS_PATH, "generic_tfidf_vectorizer.pkl")

CONTENT_DOCUMENT_TERM_MATRIX_PATH = os.path.join(MODELS_PATH, "content_document_term_matrix.pkl")
TITLE_DOCUMENT_TERM_MATRIX_PATH = os.path.join(MODELS_PATH, "title_document_term_matrix.pkl")

CONTENT_KMEANS_MODEL_PATH = os.path.join(MODELS_PATH, "content_kmeans_model.pkl")
TITLE_KMEANS_MODEL_PATH = os.path.join(MODELS_PATH, "title_kmeans_model.pkl")

class RetrievalModels:
	MAX_N_CLUSTER_TERMS = 20 # The higher the number, the higher the probability that the same terms will appear in multiple clusters

	def __init__(self):
		return

	def save_models(self):
		pickle.dump(self.dataframe, open(DATAFRAME_PATH, "wb"))
		pickle.dump(self.content_tfidf_vectorizer, open(CONTENT_TFIDF_VECTORIZER_PATH, "wb"))
		pickle.dump(self.generic_tfidf_vectorizer, open(GENERIC_TFIDF_VECTORIZER_PATH, "wb"))
		pickle.dump(self.content_document_term_matrix, open(CONTENT_DOCUMENT_TERM_MATRIX_PATH, "wb"))
		pickle.dump(self.title_document_term_matrix, open(TITLE_DOCUMENT_TERM_MATRIX_PATH, "wb"))
		pickle.dump(self.content_kmeans_model, open(CONTENT_KMEANS_MODEL_PATH, "wb"))
		pickle.dump(self.title_kmeans_model, open(TITLE_KMEANS_MODEL_PATH, "wb"))
		return

	def load_models(self):
		try:
			dataframe = pickle.load(open(DATAFRAME_PATH, "rb"))
			content_tfidf_vectorizer = pickle.load(open(CONTENT_TFIDF_VECTORIZER_PATH, "rb"))
			generic_tfidf_vectorizer = pickle.load(open(GENERIC_TFIDF_VECTORIZER_PATH, "rb"))
			content_document_term_matrix = pickle.load(open(CONTENT_DOCUMENT_TERM_MATRIX_PATH, "rb"))
			title_document_term_matrix = pickle.load(open(TITLE_DOCUMENT_TERM_MATRIX_PATH, "rb"))
			content_kmeans_model = pickle.load(open(CONTENT_KMEANS_MODEL_PATH, "rb"))
			title_kmeans_model = pickle.load(open(TITLE_KMEANS_MODEL_PATH, "rb"))

			vectorizers = (content_tfidf_vectorizer, generic_tfidf_vectorizer)
			matrices = (content_document_term_matrix, title_document_term_matrix)
			kmeans_models = (content_kmeans_model, title_kmeans_model)

			return dataframe, vectorizers, matrices, kmeans_models
		except:
			print("No existing models to load. Run the article_processing.py script")
			return None, (None, None), (None, None), (None, None)

	def set_models(self, dataframe, tfidf_vectorizers, document_term_matrices, kmeans_models):
		self.dataframe = dataframe
		self.content_tfidf_vectorizer = tfidf_vectorizers[0]
		self.generic_tfidf_vectorizer = tfidf_vectorizers[1]
		self.content_document_term_matrix = document_term_matrices[0]
		self.title_document_term_matrix = document_term_matrices[1]
		self.content_kmeans_model = kmeans_models[0]
		self.title_kmeans_model = kmeans_models[1]
		return

	def load_set_models(self):
		dataframe, tfidf_vectorizers, document_term_matrices, kmeans_models = self.load_models()
		self.set_models(dataframe, tfidf_vectorizers, document_term_matrices, kmeans_models)
		return

	def get_models(self, query_type="content"):
		if query_type == "title":
			return self.generic_tfidf_vectorizer, self.title_document_term_matrix, self.title_kmeans_model
		return self.content_tfidf_vectorizer, self.content_document_term_matrix, self.content_kmeans_model

class ArticleFetcher():
	def __init__(self, retrieval_models):
		self.retrieval_models = retrieval_models
		bags_of_words, self.content_secondary_index = self.get_top_n_cluster_terms()
		content_cluster_terms = {i: words for i, words in enumerate(bags_of_words)}
		bags_of_words, self.title_secondary_index = self.get_top_n_cluster_terms(query_type="title")
		title_cluster_terms = {i: words for i, words in enumerate(bags_of_words)}
		return

	def search_top_k_tfidf(self, query, query_type="content", top_k=5, tfidf_vectorizer=None, document_term_matrix=None, subdf=None):
		"""
		Search for the top k documents using TF-IDF using either a subset of articles or the entire corpus of articles

		Args:
				query (str): The query string searched.
				top_k (int, optional): Top k documents that should be returned. Defaults to 5.
				tfidf_vectorizer (TfidfVectorizer, optional): A TfidfVectorizer with different params than retrieval_models. Defaults to None.
				document_term_matrix (matrix, optional): A document term matrix with different weights than retrieval_models. Defaults to None.
				subdf (DataFrame, optional): The subset dataframe if any. Defaults to None.
		Returns:
				dict:
					articles (list(dict)): Article details sorted by most similar to least similar.
					total_documents_size (int): Total number of documents searched
					nonmatching_documents_size (int): Total number of documents that are not matched to the query at all
		"""
		query = preprocess_string(query, remove_punc=True)
		if query != "":
			if tfidf_vectorizer is not None and document_term_matrix is not None:
				transform_query = tfidf_vectorizer.transform([query])
				similarities = dot(transform_query, transpose(document_term_matrix))
			else:
				vectorizer, dtm, _ = self.retrieval_models.get_models(query_type)
				transform_query = vectorizer.transform([query]) # Transform the query to maintain consistency (eg. use nltk stopwords, lowercase, etc.)
				similarities = dot(transform_query, transpose(dtm))
			x = array(similarities.toarray()[0]) # Secondary_index of similarities as np array
			z = where(x == 0)[0].tolist() # The documents that didn't match with tfidf
			article_ids = argsort(x)[-top_k:][::-1] # Most similar article ids at the end -> get the last top_k using [-top_k] -> reverse the order using [::-1]
			# print(where(x == 0.0)[0].tolist())
			total_documents_size, nonmatching_documents_size = len(x), len(z)
			article_details = self.get_article_details(article_ids, subdf=subdf)
		else:
			article_details = self.get_article_details(range(top_k))
			total_documents_size = self.retrieval_models.dataframe.shape[0]
			nonmatching_documents_size = total_documents_size
		processing_info = {
			"articles": article_details,
			"total_documents_size": total_documents_size,
			"nonmatching_documents_size": nonmatching_documents_size
		}
		return processing_info

	def search_top_k_secondary_index(self, query, query_type="content", top_k=5):
		"""Search for the top k documents using the secondary index (which points to a list of cluster indexes which then point to a subset of articles) and TF-IDF on the subset of articles

		Args:
				query (str): The query string searched
				top_k (int, optional): Top k documents that should be returned. Defaults to 5.

		Returns:
				dict:
					article_details (list(dict)): Article details sorted by most similar to least similar.
					total_documents_size (int): Total number of documents searched
					nonmatching_documents_size (int): Total number of documents that are not matched to the query at all
		"""
		query = preprocess_string(query, remove_punc=True)
		if query != "":
			secondary_index = self.title_secondary_index if query_type == "title" else self.content_secondary_index
			cluster_indexes = secondary_index.get(query)
			vectorizer, _, kmeans_model = self.retrieval_models.get_models(query_type)
			if cluster_indexes is None:
				transform_query = vectorizer.transform([query])
				cluster_indexes = kmeans_model.predict(transform_query).tolist()
			if len(cluster_indexes) == 1:
				article_ids = where(kmeans_model.labels_ == cluster_indexes[0])[0].tolist()
			else:
				article_ids = set() # the order of documents is not significant in sets which is why the order can be a bit different. the order for the purpose of evaluating similarities is not relevant
				for cluster_index in cluster_indexes:
					cluster_article_ids = where(kmeans_model.labels_ == cluster_index)[0].tolist()
					article_ids.update(cluster_article_ids)
			# problem is the articles are ordered based on when they're inserted, so we'll use TF-IDF again, but with subset of articles that are clustered based on the labels. If the subset of article ids is the same as the number of records, i.e the query is predicted to be a part of all clusters (too generic), then just use the same method as it were TF-IDF
			if len(article_ids) == self.retrieval_models.dataframe.shape[0]:
				processing_info = self.search_top_k_tfidf(query, query_type, top_k)
			else:
				subdf = self.retrieval_models.dataframe.iloc[list(article_ids)]
				# Use a generic TF-IDF vectorizer which doesn't have any limiters
				column_label = "cleaned_title" if query_type == "title" else "cleaned_content"
				generic_tfidf_vectorizer = TfidfVectorizer(
					stop_words=nltk.corpus.stopwords.words("english"), # Use the NLTKs stopwords
					lowercase=True, use_idf=True, smooth_idf=True
				)
				# having to recreate the document term matrix every time makes the secondary index much slower since we still need rank based on the subset of articles
				document_term_matrix = create_document_term_matrix(generic_tfidf_vectorizer, subdf[column_label])
				# search using the same search_top_k_tfidf method
				processing_info = self.search_top_k_tfidf(query, query_type, top_k, generic_tfidf_vectorizer, document_term_matrix, subdf)
		else:
			processing_info = self.search_top_k_tfidf(query, query_type, top_k)
		return processing_info

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

	def get_top_n_cluster_terms(self, query_type="content", n=10):
		vectorizer, _, kmeans_model = self.retrieval_models.get_models(query_type)
		terms = vectorizer.get_feature_names()
		centers = kmeans_model.cluster_centers_.argsort()[:, ::-1]
		bags_of_words = [[terms[j] for j in centers[i, :n]] for i in range(NUM_CLUSTERS)] # get the closest n terms to the center of each cluster
		secondary_index = {}
		# The idea is that the "secondary index" has the words which points to a cluster index which points to a subset of articles
		for cluster_index, bag_of_words in enumerate(bags_of_words):
			for word in bag_of_words:
				if word in secondary_index:
					secondary_index[word].append(cluster_index)
				else:
					secondary_index[word] = [cluster_index]
		return bags_of_words, secondary_index

def create_document_term_matrix(tfidf_vectorizer, articles_info):
	document_term_matrix = tfidf_vectorizer.fit_transform(articles_info)
	return document_term_matrix

def create_df(path=None):
	articles_path = ARTICLES_PATH if path is None else path
	df = read_csv(articles_path, encoding="utf-8").reset_index()
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
	content_tfidf_vectorizer = TfidfVectorizer(
		max_df=0.25, # Drop words that occur in more than x% of documents -> kind of like stopwords
		min_df=5, # Drop words that occur in less than x documents -> really rare words along a large corpus doesn't help. Try to use a number proportional to the number of documents used for the models
		stop_words=nltk.corpus.stopwords.words("english"), # Use the NLTKs stopwords
		lowercase=True,
		use_idf=True,
		smooth_idf=True,
		# ngram_range=(1, 2) # (x, y): vectorization can consist of x to y words (eg. software engineering = [software, engineering] or software engineering)
	)
	generic_tfidf_vectorizer = TfidfVectorizer(
		stop_words=nltk.corpus.stopwords.words("english"), # Use the NLTKs stopwords
		lowercase=True, use_idf=True, smooth_idf=True
	)
	df = create_df()

	print("Preprocessing article features...")
	df["content"] = df["content"].apply(preprocess_string)
	df["cleaned_content"] = df["content"].apply(clean_punctuation) # Save computation time at the expense of memory by creating a new column with the cleaned data
	df["cleaned_title"] = df["title"].apply(lambda title: preprocess_string(title, remove_punc=True))

	print("Creating document term matrices...")
	content_document_term_matrix = create_document_term_matrix(content_tfidf_vectorizer, df["cleaned_content"])
	title_document_term_matrix = create_document_term_matrix(generic_tfidf_vectorizer, df["cleaned_title"])

	print("Fitting document term matrices to KMeans...")
	content_kmeans_model = MiniBatchKMeans(n_clusters=NUM_CLUSTERS)
	content_kmeans_model.fit(content_document_term_matrix)
	title_kmeans_model = MiniBatchKMeans(n_clusters=NUM_CLUSTERS)
	title_kmeans_model.fit(title_document_term_matrix)
	retrieval_models = RetrievalModels()

	print("Setting and saving models...")
	vectorizers = (content_tfidf_vectorizer, generic_tfidf_vectorizer)
	matrices = (content_document_term_matrix, title_document_term_matrix)
	kmeans_models = (content_kmeans_model, title_kmeans_model)
	retrieval_models.set_models(df, vectorizers, matrices, kmeans_models)
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
