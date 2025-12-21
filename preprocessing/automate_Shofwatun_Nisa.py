# trigger github actions
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

INPUT_PATH = "text_emotion_raw/text_emotion.csv"
OUTPUT_PATH = "preprocessing/text_emotion_preprocessing/text_emotion_clean.csv"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def preprocess():
    print("Load dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Drop NA & duplicates...")
    df = df.dropna().drop_duplicates()

    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)

    stop_words = set(stopwords.words('english'))
    df['processed_text'] = df['clean_text'].apply(
        lambda x: ' '.join(
            [w for w in word_tokenize(x) if w not in stop_words]
        )
    )

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['emotion'])

    df_final = df[['processed_text', 'label_encoded']]
    df_final.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessing SUCCESS ✅")

if __name__ == "__main__":
    preprocess()
