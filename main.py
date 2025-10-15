import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from utils import preprocess_text, plot_confusion_matrix, show_top_words

df = pd.read_csv('dataset.csv')
df['clean_review'] = df['review'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_vect, y_train)

y_pred = model.predict(X_test_vect)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred)
show_top_words(model, vectorizer, n=20)
print("Project completed successfully.")