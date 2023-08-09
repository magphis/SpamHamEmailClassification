from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

def temizleme(Veriseti):
    X_temizlenmis = []
    for eposta in Veriseti:
        try:
            # Parsing HTML
            icerik = eposta.get_content()
        except LookupError:
            # Handle encoding problem
            icerik = str(eposta.get_payload())
        except AttributeError:
            icerik = eposta


        # Cleaning emails from HTML tags   
        html_text = BeautifulSoup(icerik, 'html.parser')
        html_to_text = html_text.get_text()

        # Tokenization + Convert hyperlinks to "URL" string
        tokenized_html = []
        urls = html_text.find_all('a', href=True)
        for url in urls:
            html_to_text = html_to_text.replace(url['href'], 'URL')
            tokenized_html.append('URL')
        
        tokenized_html.extend(word_tokenize(html_to_text))

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokenized_html]

        # Convert digits to "NUM" string
        lemmatized_tokens_num = ['NUM' if token.isdigit() else token for token in lemmatized_tokens]

        # Reconvert tokens to text
        text = ' '.join(lemmatized_tokens_num)
        X_temizlenmis.append(text)

    return X_temizlenmis