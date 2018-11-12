import os
from sklearn.externals import joblib
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_url_path='')

aspects = ['produk', 'pengiriman', 'packaging']
aspect_predictor_file = [os.getenv('APP_{}_ASPECT_MODEL'.format(aspect), '{}_aspect.sav'.format(aspect)) for aspect in aspects]
sentiment_predictor_file = [os.getenv('APP_{}_SENTIMENT_MODEL'.format(aspect), '{}_sentiment.sav'.format(aspect)) for aspect in aspects]

aspect_model = list(map(lambda file: joblib.load(file), aspect_predictor_file))
sentiment_model = list(map(lambda file: joblib.load(file), sentiment_predictor_file))

@app.route('/predict/')
def predict():
    query_str = request.args.get('s')
    if query_str is None:
        return jsonify({})
    
    result = {}
    for name, m_aspect, m_sentiment in zip(aspects, aspect_model, sentiment_model):
        aspect_result = m_aspect.predict([query_str])
        if aspect_result[0] > 0:
            try:
                sentiment_result = m_sentiment.predict_proba([query_str])
            except:
                sentiment_result = m_sentiment.predict([query_str])
            result[name] = float(sentiment_result[0])

    return jsonify(result)

@app.route('/')
def hello_world():
    return send_from_directory(os.getcwd(), 'index.html')

if __name__ == '__main__':
    app.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', '3000'))
    )
