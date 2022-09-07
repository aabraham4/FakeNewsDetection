import re
import string

from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer, PorterStemmer


class FakeNewsFunctions:
    # function to change text to lowercase, remove punctuation and remove stopwords
    def remove_stopwords(self, text):
        new_text = []

        for t in range(len(text)):
            text_in_lower = text[t].lower()
            text_no_punct = text_in_lower.translate(str.maketrans('', '', string.punctuation))

            pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
            text_without_sw = pattern.sub('', text_no_punct)

            new_text.append(text_without_sw)

        return new_text

    # function to lemmatize each word in text afterr converting to tokens and return as text
    def lemmatize_text(self, text):
        new_text = []

        for t in range(len(text)):
            text_in_lower = text[t].lower()
            text_no_punct = text_in_lower.translate(str.maketrans('', '', string.punctuation))

            pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
            text_without_sw = pattern.sub('', text_no_punct)

            # tokenize text
            text_tokenized = word_tokenize(text_without_sw)

            # initilize lemmatizer
            lemmatizer = WordNetLemmatizer()
            # lemmatize each word in text where possible
            lemmatized_text = [lemmatizer.lemmatize(word) for word in text_tokenized]

            joined_text = ' '.join(lemmatized_text)
            new_text.append(joined_text)
        return new_text

    # function to stem each word in text afterr converting to tokens and return as text
    def stem_text(self, text):
        new_text = []

        for t in range(len(text)):
            text_in_lower = text[t].lower()
            text_no_punct = text_in_lower.translate(str.maketrans('', '', string.punctuation))

            pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
            text_without_sw = pattern.sub('', text_no_punct)

            # tokenize text
            text_tokenized = word_tokenize(text_without_sw)

            # initialize porter stemmer
            porter = PorterStemmer()
            # stem each word in text
            stemmed_text = [porter.stem(word) for word in text_tokenized]
            joined_text = ' '.join(stemmed_text)
            new_text.append(joined_text)
        return new_text

    # function to convert textual data into binary format.
    # True and mostly-true articles labelled as 1 and the remaining labelled as 0
    def convert_labels_to_binary(self, labels):
        new_labels = []
        for term in labels:
            if term == 'true' or term == 'mostly-true':
                new_label = 1
            else:
                new_label = 0
            new_labels.append(new_label)

        return new_labels

    # function to assign numeric value to textual labels
    # each label representing the degree of fake news from false to true from 0 to
    def convert_labels_to_numbers(self, labels):
        new_labels = []
        for term in labels:
            if term == 'pants-fire':
                new_label = 0
            elif term == 'false':
                new_label = 1
            elif term == 'barely-true':
                new_label = 2
            elif term == 'half-true':
                new_label = 3
            elif term == 'mostly-true':
                new_label = 4
            else:
                new_label = 5
            new_labels.append(new_label)

        return new_labels
