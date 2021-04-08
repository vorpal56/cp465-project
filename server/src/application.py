from flask import Flask, request
from utils import APP_PATH, SERVER_PATH, DATA_PATH, MODELS_PATH, NUM_CLUSTERS, time_call
from article_processing import ArticleFetcher, RetrievalModels

application = Flask(__name__)
application.secret_key = "doesntMatter" # Doesn't matter since we're not deploying it live
application.jinja_env.trim_blocks = True
application.jinja_env.lstrip_blocks = True

retrieval_models = RetrievalModels()
retrieval_models.load_set_models()
article_fetcher = ArticleFetcher(retrieval_models)

@application.route("/api/documents-tfidf", methods=["GET"])
@time_call
def get_tfidf_documents():
	query = request.args.get("q")
	quantity = request.args.get("n")
	query_type = request.args.get("t")
	if quantity is not None:
		article_objects = article_fetcher.search_top_k_tfidf(query, query_type=query_type, top_k=int(quantity))
	else:
		article_objects = article_fetcher.search_top_k_tfidf(query, query_type=query_type)

	return article_objects

@application.route("/api/documents-si", methods=["GET"])
@time_call
def get_secondary_index_documents():
	query = request.args.get("q")
	quantity = request.args.get("n")
	query_type = request.args.get("t")
	if quantity is not None:
		article_objects = article_fetcher.search_top_k_secondary_index(query, query_type=query_type, top_k=int(quantity))
	else:
		article_objects = article_fetcher.search_top_k_secondary_index(query, query_type=query_type)
	return article_objects

if __name__ == "__main__":
	application.run(host="0.0.0.0", port=5000, debug=True)
