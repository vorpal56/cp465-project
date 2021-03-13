## Getting Started
You will need [Node.js](https://nodejs.org/en/) to run this project, [npm](https://www.npmjs.com/) (which comes with Node.js during installation), and [Angular 10+](https://angular.io/). To update data, you will need [Python 3](https://docs.python.org/3/) as well.
1. `git clone https://github.com/vorpal56/cp465-project.git`
2. `cd cp465-project/`
3. `npm install -g @angular/cli && npm install`

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
Served on port `5000`, routed in `proxy.json` as `/api` to port `4200`, the default port for Angular applications
```
npm run py-dev
```
