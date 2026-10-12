"""Text preprocessing for lexical and semantic retrieval."""
import re
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

# Words to retain even though they're in standard stop-word lists
RETAIN_WORDS = {
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whom',
    'not', 'no', 'never', 'neither', 'nor',
    'can', 'could', 'should', 'would', 'will',
    'between', 'against', 'above', 'below',
}

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "i'm": "i am", "i've": "i have",
    "i'll": "i will", "i'd": "i would", "you're": "you are",
    "you've": "you have", "you'll": "you will", "he's": "he is",
    "she's": "she is", "it's": "it is", "we're": "we are",
    "they're": "they are", "that's": "that is", "what's": "what is",
    "who's": "who is", "there's": "there is",
}

# Lazy loading of nltk resources
_stemmer = None
_stop_words = None


def _get_stemmer_and_stopwords():
    global _stemmer, _stop_words
    if _stemmer is None or _stop_words is None:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        _stemmer = PorterStemmer()
        _stop_words = set(stopwords.words('english')) - RETAIN_WORDS
    return _stemmer, _stop_words


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    apply_stemming: bool = True,
    return_tokens: bool = False
) -> Union[str, List[str]]:
    """Preprocess text for lexical retrieval (TF-IDF and BM25).

    Args:
        text: Raw question text.
        remove_stopwords: Whether to remove non-informative stop words.
        apply_stemming: Whether to apply Porter stemming.
        return_tokens: If True, return list of tokens; else joined string.

    Returns:
        Preprocessed text string or list of tokens.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return [] if return_tokens else ""

    # 1. Lowercase
    text = text.lower()

    # 2. Expand common contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    # 3. Clean special characters (preserve alphanumeric, spaces, hyphens)
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)

    # 4. Tokenize
    stemmer, stop_words = _get_stemmer_and_stopwords()
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)

    # 5. Selective stop-word removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # 6. Porter stemming
    if apply_stemming:
        tokens = [stemmer.stem(t) for t in tokens]

    if return_tokens:
        return tokens
    return ' '.join(tokens)


def preprocess_for_sbert(text: str) -> str:
    """Minimal preprocessing for SBERT - preserves original casing and punctuation.

    SBERT has its own WordPiece / BPE tokenizer and is trained on natural
    syntax, capitalization, and punctuation.
    """
    if not isinstance(text, str):
        return ""
    return text.strip()
