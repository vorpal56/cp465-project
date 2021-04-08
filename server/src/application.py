from flask import Flask, session, request, render_template, url_for, redirect
from utils import APP_PATH, SERVER_PATH, DATA_PATH, MODELS_PATH, NUM_CLUSTERS, create_response
from article_processing import ArticleFetcher, RetrievalModels
import json

application = Flask(__name__)
application.secret_key = "doesntMatter"
application.jinja_env.trim_blocks = True
application.jinja_env.lstrip_blocks = True

retrieval_models = RetrievalModels()
retrieval_models.load_set_models()
article_fetcher = ArticleFetcher(retrieval_models)

@application.route("/api/documents-tfidf", methods=["GET"])
def get_tfidf_documents():
	query = request.args.get("q")
	count = request.args.get("n")
	query_type = request.args.get("type")
	if count is not None:
		article_objects = article_fetcher.search_top_k_tfidf(query, top_k=int(count))
	else:
		article_objects = article_fetcher.search_top_k_tfidf(query)
	return json.dumps(article_objects)


@application.route("/api/documents-si", methods=["GET"])
def get_secondary_index_documents():
	query = request.args.get("q")
	count = request.args.get("n")
	query_type = request.args.get("type")
	if count is not None:
		article_objects = article_fetcher.search_top_k_secondary_index(query, top_k=int(count))
	else:
		article_objects = article_fetcher.search_top_k_secondary_index(query)
	return json.dumps(article_objects)

if __name__ == "__main__":
	application.run(host="0.0.0.0", port=5000, debug=True)
