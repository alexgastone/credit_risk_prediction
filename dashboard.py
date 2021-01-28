from explainerdashboard import ClassifierExplainer, ExplainerDashboard

explainer = ClassifierExplainer.from_file("explainer.joblib")
db = ExplainerDashboard(explainer)

app = db.flask_server()

def main():
    app.run(host='0.0.0.0', port=4444, debug=True)

if __name__ == '__main__':
    main()

#$ gunicorn -w 3 --preload -b localhost:8050 dashboard:app
#$gunicorn --preload dashboard:app