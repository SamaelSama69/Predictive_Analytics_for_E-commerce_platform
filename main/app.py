from flask import Flask, render_template, request
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment_analyzer import SentimentAnalyzer
from clustering_models import ClusteringModels
from delivery_analyzer import DeliveryAnalyzer

app = Flask(__name__)

# Global objects
sentiment_analyzer_obj = None
clustering_models_obj = None
delivery_analyzer_obj = None


def load_models():
    global sentiment_analyzer_obj, clustering_models_obj, delivery_analyzer_obj

    print("Loading models...")

    # Sentiment
    try:
        sentiment_analyzer_obj = SentimentAnalyzer(
            s3_bucket_name='ecom-models-007',
            s3_model_key_prefix=''
        )
    except:
        sentiment_analyzer_obj = SentimentAnalyzer(
            model_name_or_path="./sentiment_model"
        )

    # Clustering
    try:
        clustering_models_obj = ClusteringModels(
            seller_model_path="models/seller_clustering_model.pkl",
            review_model_path="models/review_clustering_model.pkl",
            customer_model_path="models/customer_clustering_model.pkl"
        )
    except:
        clustering_models_obj = None

    # Delivery
    try:
        delivery_analyzer_obj = DeliveryAnalyzer(None)
    except:
        delivery_analyzer_obj = None


# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    result = None

    if request.method == 'POST':
        text = request.form.get('review_text')

        if sentiment_analyzer_obj:
            result = sentiment_analyzer_obj.analyze_sentiment(text)

    return render_template('sentiment.html', result=result)


@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    result = None

    if request.method == 'POST':
        if clustering_models_obj:
            features = [
                float(request.form.get('f1', 0)),
                float(request.form.get('f2', 0)),
                float(request.form.get('f3', 0))
            ]
            result = clustering_models_obj.predict_customer_segment(features)

    return render_template('clustering.html', result=result)


@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting():
    result = None

    if request.method == 'POST':
        periods = int(request.form.get('periods'))
        forecast = [(f"Month {i+1}", 5000 + i * 200) for i in range(periods)]
        result = forecast

    return render_template('forecasting.html', result=result)


# ---------------- RUN ---------------- #

if __name__ == '__main__':
    load_models()
    app.run(debug=True)