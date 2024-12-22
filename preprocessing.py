import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import LabelEncoder
import unicodedata
import contractions
from collections import Counter
import logging


class AdvancedPreprocessor:
    def __init__(self, input_file, output_dir="data/",
                 min_length=3, max_length=100,
                 test_size=0.1, val_size=0.1,
                 random_state=42):
        self.input_file = input_file
        self.output_dir = output_dir
        self.min_length = min_length
        self.max_length = max_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words_en = set(stopwords.words('english'))
        self.stop_words_de = set(stopwords.words('german'))

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        """Load and perform initial data validation"""
        try:
            self.df = pd.read_csv(self.input_file, delimiter="\t", usecols=[1, 3], names=["en", "de"], quoting=3,
                                  header=None)
            self.logger.info(f"Loaded {len(self.df)} sentence pairs.")
            return True
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False

    def clean_text(self, text, is_english=True):
        """Clean text for English or German"""
        text = text.lower()

        if not is_english:
            text = unicodedata.normalize('NFKC', text)
        else:
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

        if is_english:
            text = contractions.fix(text)

        text = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', '', text if is_english else text)

        text = ' '.join(text.split())
        return text

    def filter_sentences(self):
        """Filter sentences based on length and content criteria"""
        self.df = self.df.dropna()

        self.df['en'] = self.df['en'].astype(str)
        self.df['de'] = self.df['de'].astype(str)

        length_mask = (
                (self.df['en'].str.len() >= self.min_length) &
                (self.df['de'].str.len() >= self.min_length) &
                (self.df['en'].str.len() <= self.max_length) &
                (self.df['de'].str.len() <= self.max_length)
        )
        self.df = self.df[length_mask]

        self.df = self.df.drop_duplicates()

        self.logger.info(f"After filtering: {len(self.df)} sentence pairs remain.")

    def perform_advanced_cleaning(self):
        """Apply advanced cleaning to both languages"""
        self.df['en_cleaned'] = self.df['en'].apply(lambda x: self.clean_text(str(x), is_english=True))
        self.df['de_cleaned'] = self.df['de'].apply(lambda x: self.clean_text(str(x), is_english=False))

        self.logger.info(f"Sample cleaned English: {self.df['en_cleaned'].head()}")
        self.logger.info(f"Sample cleaned German: {self.df['de_cleaned'].head()}")

        self.calculate_text_statistics()

    def calculate_text_statistics(self):
        """Calculate and log various text statistics"""
        if 'en_cleaned' not in self.df.columns or 'de_cleaned' not in self.df.columns:
            self.logger.error("Cleaned columns not found. Ensure cleaning is complete.")
            return

        en_words = ' '.join(self.df['en_cleaned']).split()
        de_words = ' '.join(self.df['de_cleaned']).split()

        en_vocab = Counter(en_words)
        de_vocab = Counter(de_words)

        self.logger.info(f"Top English words: {en_vocab.most_common(10)}")
        self.logger.info(f"Top German words: {de_vocab.most_common(10)}")

        self.logger.info(f"English vocabulary size: {len(en_vocab)}")
        self.logger.info(f"German vocabulary size: {len(de_vocab)}")
        self.logger.info(
            f"Average English sentence length: {np.mean([len(x.split()) for x in self.df['en_cleaned']]):.2f} words")
        self.logger.info(
            f"Average German sentence length: {np.mean([len(x.split()) for x in self.df['de_cleaned']]):.2f} words")

    def split_data(self):
        """Split data into train, validation, and test sets"""
        data = self.df[['en_cleaned', 'de_cleaned']]

        train_val, test = train_test_split(data, test_size=self.test_size, random_state=self.random_state)

        val_size_adjusted = self.val_size / (1 - self.test_size)
        train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=self.random_state)

        self.logger.info(f"Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}")
        return train, val, test

    def save_data(self, train, val, test):
        """Save processed datasets with vocabularies"""
        train.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(self.output_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)

        vocab_en = Counter(' '.join(train['en_cleaned']).split())
        vocab_de = Counter(' '.join(train['de_cleaned']).split())

        pd.DataFrame(vocab_en.most_common(), columns=['word', 'count']).to_csv(
            os.path.join(self.output_dir, "vocab_en.csv"), index=False)
        pd.DataFrame(vocab_de.most_common(), columns=['word', 'count']).to_csv(
            os.path.join(self.output_dir, "vocab_de.csv"), index=False)

        self.logger.info("Saved datasets and vocabularies.")

    def process(self):
        """Main processing pipeline"""
        if not self.load_data():
            return False

        self.filter_sentences()
        self.perform_advanced_cleaning()
        train, val, test = self.split_data()
        self.save_data(train, val, test)
        return True


def main():
    preprocessor = AdvancedPreprocessor(
        input_file="./data/ge-en.tsv",
        output_dir="data/processed/",
        min_length=3,
        max_length=100,
        test_size=0.1,
        val_size=0.1,
        random_state=42
    )
    preprocessor.process()


if __name__ == "__main__":
    main()