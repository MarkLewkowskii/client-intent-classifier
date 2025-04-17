import joblib

clf = joblib.load("model/model.pkl")
vec = joblib.load("model/vectorizer.pkl")

def classify(text):
    vec_text = vec.transform([text])
    return clf.predict(vec_text)[0]

if __name__ == "__main__":
    example = input("Wpisz treść wiadomości: ")
    print("Kategoria:", classify(example))
