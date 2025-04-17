
```markdown
# 🧠 Client Intent Classifier

A lightweight machine learning project that classifies customer support messages into predefined intent categories such as `return`, `complaint`, `invoice inquiry`, and more.

Built with Python, `scikit-learn`, and `TfidfVectorizer` — designed for quick local predictions and easy customization for specific business needs.

---

## 📂 Categories

The model recognizes the following intent categories:
- `zwrot` – return
- `reklamacja` – complaint
- `pytanie o zamówienie` – order inquiry
- `pytanie o fakturę` – invoice inquiry
- `niezwiązane z produktem` – unrelated
- `inne pytanie` – other question

---

## 🚀 How to use

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/client-intent-classifier.git
cd client-intent-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```

This will create two files:
- `model/model.pkl` – trained classifier
- `model/vectorizer.pkl` – text vectorizer

### 4. Make predictions
```bash
python predict.py
```

Or use `classify(text)` inside your code.

---

## 🌐 Optional: Streamlit App

You can run a simple web interface using:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://client-intent-classifier-qyzbj2jw2rsbn5hzsh34eb.streamlit.app/)


---

## 📊 Dataset

The training dataset is available in:
```
data/boc_intents_dataset.csv
```

It contains 200+ artificial Polish-language messages balanced across 6 categories.  
You can replace this with real data and re-train the model easily.

---

## 🧪 Example

**Input:**
```
Chciałbym otrzymać fakturę za zamówienie.
```

**Output:**
```
pytanie o fakturę
```

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

Feel free to use or adapt for educational and non-commercial use.  
For commercial licensing – contact the author.

---

## ✨ Author

Created by Maryna Dudik 
*Junior Data Scientist & Automation Enthusiast*

```
