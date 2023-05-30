import csv
import pandas as pd
import string
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from Singkatan.SingkatanConverter import SingkatanConverter


def sentiment_analysis(input_string):
    class PerbandinganSentimentAnalyzer():
        def __init__(self, sentiment_file, singkatan_file):
            self.sentiment_dict = {}
            self.singkatan_converter = SingkatanConverter()
            self.singkatan_converter.importDictionary(singkatan_file)
            
            # Load sentiment dictionary from file
            with open(sentiment_file, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    self.sentiment_dict[row[0]] = int(row[1])

        def convert_singkatan(self, sentence):
        
            converter = SingkatanConverter()
            
            converter.importDictionary(singkatan_file)
            tokens = sentence.split() 
            converted = [converter.convert(token) for token in tokens]
            return ' '.join(converted)

        def tag_removal(self, sentence):
            # Menghapus semua link pada teks dengan regular expression
            text = re.sub(r'http\S+', '', sentence)
            return text

        def analyze_sentiment(self, sentence):
            # sentence = convert_singkatan(sentence)
            max_sentiment = 0
            min_sentiment = 0
            final_score = 0
            result = ""
            sentiment_detail = []
            prev_word = None
            words = sentence.lower().split()
            for i in range(len(words)):
                word = words[i]
                if word in self.sentiment_dict:
                    score = self.sentiment_dict[word]
                    if prev_word in ["belum", "bukan", "tak", "tanpa", "tidak", "pantang", "jangan", "bukanlah", "sok", "tidak pernah"]:
                        score = -1 * score  # Invert sentiment score after negation word
                    if score < min_sentiment:
                        min_sentiment = score
        
                    if score > max_sentiment:
                        max_sentiment = score
                    
                    sentiment_detail.append({"kata": word, "nilai": score, "sentimen": "positive" if score > 0 else "negative"})

                else:
                    sentiment_detail.append({"kata": word, "nilai": 0, "sentimen": "neutral"})
                prev_word = word
            
            if final_score == 0:
                result = "neutral"
            elif final_score > 0:
                result ="positive"
            else: 
                result ="negative"
            
            if abs(min_sentiment) > abs(max_sentiment):
                final_score = min_sentiment
                result = "negative"
                return {"sentiment": result,  "max_sentiment":max_sentiment, "min_sentiment":min_sentiment, "final_score":final_score, "detail": sentiment_detail, "negative": negative, "positive": positive}
            elif abs(max_sentiment) > abs(min_sentiment):
                final_score = max_sentiment
                result = "positive"
                return {"sentiment": result, "score": 0, "max_sentiment":max_sentiment, "min_sentiment":min_sentiment, "final_score":final_score, "detail": sentiment_detail,"negative": negative, "positive": positive}
            elif abs(min_sentiment) == abs(max_sentiment):
                final_score = 0
                result = "neutral"
                return {"sentiment": result, "score": 0, "max_sentiment":max_sentiment, "min_sentiment":min_sentiment, "final_score":final_score, "detail": sentiment_detail, "negative": negative, "positive": positive}
            

    # Load sentiment dictionary from CSV file
    sentiment_file = "SentimentCorpus2.csv"
    singkatan_file = "singkatan.csv"

    positive = 0
    negative = 0
    perbandingan_analyzer = PerbandinganSentimentAnalyzer(sentiment_file, singkatan_file)


    sentence = input_string
    sentence = sentence.lower()
    sentence = perbandingan_analyzer.convert_singkatan(sentence)
    sentence = perbandingan_analyzer.tag_removal(sentence)
    result = perbandingan_analyzer.analyze_sentiment(sentence)

    # Preprocessing sentence
    # sentence = convert_singkatan(sentence)
    print("Kalimat: {}".format(sentence))
    print("Sentimen: {}".format(result["sentiment"]))
    print("Max Sentiment: {}".format(result["max_sentiment"]))
    print("Min Sentiment: {}".format(result["min_sentiment"]))
    print("Nilai sentimen: {}".format(result["final_score"]), "\n")

    print("Detail sentimen:")
    sentiment_details = []
    for detail in result["detail"]:
        if detail["nilai"] > 0:
            positive+=1
        elif detail["nilai"] < 0:
            negative+=1
        sentiment_details.append({"kata": detail["kata"], "nilai": detail["nilai"], "sentimen": detail["sentimen"]})
        print("- kata: {}, nilai: {}, sentimen: {}".format(detail["kata"], detail["nilai"], detail["sentimen"]))

    print("\nJumlah Kata Negatif: ",negative)
    print("Jumlah Kata Positif: ",positive)

    data = pd.read_csv('FakeNewsDatatestMinMaxNormalization.csv')
    data = data.dropna()

    X = data[['max', 'min']].values
    y = data['label'].values

    
    # membuat objek Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(kernel='linear', random_state=42)
    nb = GaussianNB()

    # melatih model pada seluruh data
    rf.fit(X, y)
    svm.fit(X,y)
    nb.fit(X, y)

    #Min Max Scaler
    max_data = result["max_sentiment"]
    min_data = result["min_sentiment"]

    min_val = -5
    max_val = 5

    scaled_max = ((max_data - min_val) / (max_val - min_val)) * 2 - 1
    scaled_min = ((min_data - min_val) / (max_val - min_val)) * 2 - 1  

    print("Max Sentiment: ", max_data)
    print("Min Sentiment: ", min_data)

    print("\n Hasil Min Max Scaler")
    print("Max Scaler: ", scaled_max)
    print("Min Scaler: ", scaled_min)

    # membuat prediksi
    rf_predict = rf.predict([[scaled_max, scaled_min]])
    svm_predict = svm.predict([[scaled_max, scaled_min]])
    nb_predict = nb.predict([[scaled_max, scaled_min]])

    # menampilkan prediksi
    if rf_predict[0] == 0:
        print("Prediksi: label 0")
        rf_label = "Berita Fake"
    else:
        print("Prediksi: label 1")
        rf_label = "Berita True"

    if svm_predict[0] == 0:
        print("Prediksi: label 0")
        svm_label = "Berita Fake"
    else:
        print("Prediksi: label 1")
        svm_label = "Berita True"

    if nb_predict[0] == 0:
        print("Prediksi: label 0")
        nb_label = "Berita Fake"
    else:
        print("Prediksi: label 1")
        nb_label = "Berita True"
        
    return format(sentence), format(result["sentiment"]),format(result["max_sentiment"]),format(result["min_sentiment"]),format(result["final_score"]),negative,positive, sentiment_details,rf_label,svm_label,nb_label


