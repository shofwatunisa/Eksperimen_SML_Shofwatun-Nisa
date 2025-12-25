import pandas as pd
import re
import string
import nltk
import os

from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

# ======================
# Download NLTK resources
# ======================
nltk.download('stopwords')

# ======================
# Path input & output
# ======================
INPUT_PATH = "text_emotion_raw/text_emotion.csv"
OUTPUT_PATH = "preprocessing/text_emotion_preprocessing/text_emotion_clean.csv"

# ======================
# Text cleaning function
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# ======================
# Preprocessing pipeline
# ======================
def preprocess():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Drop missing values & duplicates...")
    df = df.dropna().drop_duplicates()

    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)

    print("Remove stopwords...")
    stop_words = set(stopwords.words('english'))
    df['processed_text'] = df['clean_text'].apply(
        lambda x: ' '.join([w for w in x.split() if w not in stop_words])
    )

    print("Encode labels...")
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['emotion'])

    df_final = df[['processed_text', 'label_encoded']]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df_final.to_csv(OUTPUT_PATH, index=False)
    print("Preprocessing SUCCESS")

# ======================
# Run script
# ======================
if __name__ == "__main__":
    preprocess()
