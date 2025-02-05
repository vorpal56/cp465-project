[Dataset available on Kaggle](https://www.kaggle.com/snapcrack/all-the-news)

## Getting Started
You will need [Node.js](https://nodejs.org/en/) to run this project, [npm](https://www.npmjs.com/) (which comes with Node.js during installation), and [Angular 10+](https://angular.io/). To update data, you will need [Python 3](https://docs.python.org/3/) as well.
1. `git clone https://github.com/vorpal56/cp465-project.git`
2. `cd cp465-project/`
3. `npm install -g @angular/cli`
4. `npm install`

It is recommended that you use [virtual environments](https://docs.python.org/3/tutorial/venv.html) which are used to isolate requirements into global and local scopes since there are many different package dependencies for data processing and serving.
### Windows
1. `python -m venv venv`
2. `./venv/Scripts/activate.bat`
3. `pip install -r requirements.txt`

### Ubuntu/Linux
1. `python3 -m venv venv`
2. `source venv/scripts/activate`
3. `pip install -r requirements.txt`

## Development Server
### Frontend
```
npm run serve
```
### Backend
The backend requires 7 models in the [`server/data/models/`](server/data/models/) folder. Since GitHub has a limit on the size of files and the number of articles we're using is large, we can't publish it here. You can create the models using `python server/src/article_processing.py` after you've installed all the Python requirements: 
1. `dataframe.pkl` - DataFrame that holds the article details from a given article path
2. `content_tfidf_vectorizer.pkl` - The TfidfVectorizer class with tuned parameters for article contents
3. `content_document_term_matrix.pkl` - The document term matrix fitted by the `content_tfidf_vectorizer` along all articles contents from the dataframe
4. `content_kmeans_model.pkl` - The KMeans model fitted by the `content_document_term_matrix`
5. `generic_tfidf_vectorizer.pkl` - The TfidfVectorizer class with generic parameters for article titles
6. `title_document_term_matrix.pkl` - The document term matrix fitted by the `generic_tfidif_vectorizer` along all articles titles from the dataframe
7. `title_kmeans_model.pkl` - The KMeans model fitted by the `title_document_term_matrix`

Served on port `5000`, routed in `proxy.json` as `/api` to port `4200`, the default port for Angular applications. 
```
cd server/
python src/application.py
```
