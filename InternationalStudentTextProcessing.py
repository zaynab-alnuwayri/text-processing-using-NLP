import os
import re
import spacy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer


class TextProcessing:
    """A preprocessor for text to perform NLP
    
    Attributes:
    --------------
    - text (str): a unit of text being processed
    - _spell_checker (dict): dictionary of common misspellings
    - _conversion_dictionary (dict): dictionary of contractions
    - self.lemmatize (bool): indicating if user wants lemmatization 
                    (if not, stemming is used)
    """

    def __init__(self, lemmatization=False, additional_stop_words=set()):
        """Initialize the text processor."""

        self.use_lemmatization = lemmatization
        self.text = ""

        # take in the data from contractions.csv if available
        if os.path.exists("contractions.csv"):
            df = pandas.read_csv("contractions.csv", index_col=0)
            self._conversion_dictionary = df.to_dict()['Unnamed: 1']
        else:
            self._conversion_dictionary = {
                "can't": "cannot",
                "won't": "will not",
                "don't": "do not",
                "didn't": "did not",
                "isn't": "is not",
                "aren't": "are not",
                "i'm": "i am",
                "it's": "it is",
                "that's": "that is",
                "they're": "they are",
                "i've": "i have",
                "you're": "you are",
            }

        # take data from aspell.txt as dictionary
        self._spell_checker = dict()
        with open("aspell.txt", encoding="utf-8") as aspell:
            for line in aspell:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                line = line.split(":")
                value = line[0]
                keys = line[1].split()
                line_dict = {key: value for key in keys}
                self._spell_checker.update(line_dict)

         # Initialize spaCy model for lemmatization and POS tagging
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize for stemming 
        from nltk.stem import PorterStemmer
        self.stemmer = PorterStemmer()
    

        # Define a basic list of English stopwords
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }

        self.stop_words.update(additional_stop_words)

    def set_text(self, text: str):
        """Set and process the text"""
        self.text = str(text)
        self.fix_contractions()
        self.cleaning()
        self.remove_extra_whitespace()
        self.tokenize()
        self.fix_spelling()
        self.remove_numbers()
        self.remove_stopwords()

        if self.use_lemmatization:
            self.lemmatize_text()
        else:
            self.stem_text()

        return self.text

    def fix_spelling(self):
        """Eliminate some misspellings using aspell.txt."""
        if not isinstance(self.text, list):
            self.tokenize()

        self.text = [self._spell_checker.get(word, word) for word in self.text]

    def fix_contractions(self):
        """Eliminate contractions using the contraction dictionary."""
        if isinstance(self.text, str):
            words = self.text.split()
            new_words = []
            for word in words:
                lower_word = word.lower()
                if lower_word in self._conversion_dictionary:
                    word = self._conversion_dictionary[lower_word]
                new_words.append(word)
            self.text = " ".join(new_words)
        else:
            print("text is not a string.")

    def cleaning(self):
        """Mutate text to be lower case and remove noise."""
        if isinstance(self.text, str):
            self.text = self.text.lower()
            self.text = re.sub(r'<[^>]+>', ' ', self.text)
            self.text = re.sub(r'http\S+|www\.\S+', ' ', self.text)
            self.text = ''.join(
                char for char in self.text
                if char.isspace() or char.isalpha()
            )
        else:
            raise TypeError("Text must be a string for cleaning.")

    def remove_stopwords(self):
        """Remove stopwords from the text."""
        if not isinstance(self.text, list):
            self.tokenize()

        self.text = [word for word in self.text if word not in self.stop_words]

    def remove_numbers(self):
        """Remove numbers from the text."""
        if not isinstance(self.text, list):
            self.tokenize()

        self.text = [word for word in self.text if not word.isdigit()]

    def remove_extra_whitespace(self):
        """Remove extra whitespace from the text."""
        if isinstance(self.text, str):
            self.text = ' '.join(self.text.split())
        elif isinstance(self.text, list):
            self.text = [word for word in self.text if word.strip()]

    def tokenize(self, delimiter=" "):
        """Tokenize the text into words.

        Args:
            delimiter (str): The delimiter to split the text on.
        """
        if isinstance(self.text, str):
            self.text = self.text.lower()
            self.text = self.text.split(delimiter)
            self.text = [word for word in self.text if word]
        else:
            raise TypeError(
                f"Text must be a string, but it is a {type(self.text)}"
            )

    def lemmatize_text(self):
        """Lemmatize the text using spaCy."""
        if not isinstance(self.text, list):
            self.tokenize()
        doc = self.nlp(" ".join(self.text))
        lemmas = []
        for token in doc:
            lemma = token.lemma_ if token.lemma_ else token.text
            if lemma == "-PRON-":
                lemma = token.text
            lemmas.append(lemma)
        self.text = lemmas

    def stem_text(self):
        """Stem text using Porter Stemmer."""
        if not isinstance(self.text, list):
            self.tokenize()

        self.text = [self.stemmer.stem(word) for word in self.text]

    def pos_tag_text(self):
        """Add POS tags using spaCy."""
        if not isinstance(self.text, list):
            self.tokenize()

        doc = self.nlp(" ".join(self.text))
        return [(token.text, token.pos_) for token in doc]

    def process_dataframe(self, df, text_column="Text"):
        """Process a dataframe column and return a copy with cleaned tokens."""
        new_df = df.copy()
        new_df["processed_text"] = new_df[text_column].apply(self.set_text)
        new_df["processed_text_string"] = new_df["processed_text"].apply(
            lambda tokens: " ".join(tokens)
        )
        return new_df

    def vectorize(self, texts):
        """Convert processed texts into TF-IDF vectors."""
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        return matrix, vectorizer
