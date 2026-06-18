# 🧠 Neuro-Fuzzy Text Sentiment Analyzer  
### Hybrid AI System: Logistic Regression + Fuzzy Logic

This project implements a Neuro-Fuzzy Sentiment Analysis system that blends machine-learning probability outputs with fuzzy logic–based human-interpretable sentiment levels.  
The Streamlit app provides a modern dark-themed UI, probability metrics, fuzzy membership graphs, and real-time sentiment analysis.

---

## 🚀 Features
- **Neural Layer**
  - TF-IDF vectorizer  
  - Logistic Regression classifier  
  - Produces probability *p ∈ [0,1]* for positive sentiment  

- **Fuzzy Logic Layer**
  - Five triangular membership functions  
  - Categories:
    - Strong Negative  
    - Somewhat Negative  
    - Neutral  
    - Somewhat Positive  
    - Strong Positive  
  - Max-membership decision + tie-break priority  

- **Streamlit Web App**
  - Premium dark UI  
  - Animated background  
  - Example text buttons  
  - Membership distribution visualizations  
  - Probability + fuzzy label + confidence score  

---

## 📂 Project Structure
```

.
├── app.py                  # Streamlit application
├── sentiment_model.pkl     # Saved Logistic Regression model
├── NFTSA.ipynb             # Training + fuzzy logic notebook
├── data/                   # Dataset folder (download link below)
├── README.md

```

---

## 📦 Dataset Information

This project uses the **IMDB Large Movie Review Dataset** created by Maas et al. (Stanford AI Lab), a widely used benchmark for sentiment analysis.

### **Dataset Details**
- 50,000 movie reviews  
- Balanced dataset: 25k positive / 25k negative  
- Pre-labeled for binary sentiment classification  
- Used to train:
  - TF-IDF vectorizer  
  - Logistic Regression sentiment classifier  

### **Why This Dataset?**
- High-quality human-written reviews  
- Strong polarity indicators  
- Standard benchmark → reproducible performance  
- Perfect for baseline ML + fuzzy logic hybrid systems  

### **Download Link (Official Source)**
The dataset is too large to store in GitHub (Git LFS not used here).  
Download it manually from the official link:

🔗 **IMDB Dataset:**  
http://ai.stanford.edu/~amaas/data/sentiment/

After downloading, place files in:

```

data/

```

Example:
```

data/IMDB Dataset.csv.zip

````

---

## 🛠 Installation

### Clone Repo
```bash
git clone https://github.com/sampathmagapu/Neuro-Fuzzy-Text-Sentiment-Analyzer.git
cd Neuro-Fuzzy-Text-Sentiment-Analyzer
````

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit scikit-learn matplotlib numpy
```

### Run App

```bash
streamlit run app.py
```

> Make sure `sentiment_model.pkl` is in the same folder.
> If missing, run the training notebook to generate it.

---

## 📊 How the System Works

### 1️⃣ Machine Learning (Neural Layer)

* Clean text
* Convert to TF-IDF features
* Logistic Regression outputs *p = probability of positive sentiment*

### 2️⃣ Fuzzy Logic Interpretation

Triangular membership functions:

| Class      | (a, b, c)          |
| ---------- | ------------------ |
| strong_neg | (0.00, 0.00, 0.20) |
| some_neg   | (0.10, 0.30, 0.45) |
| neutral    | (0.35, 0.50, 0.65) |
| some_pos   | (0.55, 0.70, 0.85) |
| strong_pos | (0.80, 1.00, 1.00) |

Each outputs a membership μ ∈ [0,1].

### 3️⃣ Final Output

* System selects class with **highest membership**
* Returns:

  * Neural probability
  * Fuzzy label
  * Confidence
  * Membership bars & visualizations

---

## 📈 Example Predictions

| Text                            | p    | Sentiment       |
| ------------------------------- | ---- | --------------- |
| “Absolutely fantastic movie!”   | 0.89 | Strong Positive |
| “The story was dull and slow.”  | 0.12 | Strong Negative |
| “It was okay, nothing special.” | 0.51 | Neutral         |

---

## 🔮 Future Improvements

* Add BERT embeddings
* Add ANFIS (adaptive neuro-fuzzy inference)
* PDF export
* Multi-language support

---

## 👤 Author

**Anil Sai Vallabhu**  
Email: [vallabhuanilsai@gmail.com](mailto:vallabhuanilsai@gmail.com)  
LinkedIn: [https://www.linkedin.com/in/vallabhu-anil-sai-70a5a1366/](https://www.linkedin.com/in/vallabhu-anil-sai-70a5a1366/)

---

## 📄 License

MIT License

```
