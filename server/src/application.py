from utils import APP_PATH, SERVER_PATH, DATA_PATH, MODELS_PATH, create_response
from flask import Flask, session, request, render_template, url_for, redirect

application = Flask(__name__)
application.secret_key = "doesntMatter"
application.jinja_env.trim_blocks = True
application.jinja_env.lstrip_blocks = True

@application.route("/document", methods=["GET"])
def get_documents():
  documents = ["doc1", "doc2", "doc3"]
  return create_response(200, str(documents))

if __name__ == "__main__":
  application.run(host="0.0.0.0", port=5000, debug=True)
