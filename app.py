from flask import Flask, render_template,request
from casefolding import case_folding
from DocumentSimilarity import document_similarity
from SentimentAnalysis import sentiment_analysis

app = Flask(__name__)


@app.route("/")
def indexku():
    return render_template("home.html")


@app.route('/fake-news-detection', methods=['POST'])
def fake_news_detection_handler():
  result = []
  input_string = request.form['input_string']
  # sentiment_details = result["detail"]
  output_string = document_similarity(input_string)
  sentiment_output = sentiment_analysis(input_string)
  return render_template('result.html', output_string=output_string, sentiment_output=sentiment_output)

if __name__ == "__main__":
    app.run(debug=True)
