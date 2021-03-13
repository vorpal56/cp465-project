from flask import Flask, session, request, render_template, url_for, redirect
from utils import APP_PATH, SERVER_PATH, DATA_PATH, MODELS_PATH, create_response
from article_tfidf import ArticleFetcher, TfidfModels
import json

application = Flask(__name__)
application.secret_key = "doesntMatter"
application.jinja_env.trim_blocks = True
application.jinja_env.lstrip_blocks = True

tfidf_model = TfidfModels()
tfidf_model.load_set_models()
article_fetcher = ArticleFetcher(tfidf_model)

@application.route("/document", methods=["GET"])
def get_documents():
	query = request.args.get("q")
	count = request.args.get("n")
	if count is not None:
		article_objects = article_fetcher.search_top_k(query, top_k=count)
	else:
		article_objects = article_fetcher.search_top_k(query)
	return json.dumps(article_objects)

if __name__ == "__main__":
	application.run(host="0.0.0.0", port=5000, debug=True)
