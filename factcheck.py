# factcheck.py

import torch
import torch.nn.functional as F
from typing import List, Dict, Set
import numpy as np
import spacy
import gc
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('stopwords')
ps = PorterStemmer()

class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)

        # Assuming the output order is ["entailment", "neutral", "contradiction"]
        entailment_score = probabilities[0, 0].item()

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        return entailment_score


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):
    def __init__(self, threshold=0.034):
        # Threshold for similarity to decide whether fact is supported or not
        self.threshold = threshold
        # Set of stopwords for the tokenizer to remove
        self.stop_words = set(stopwords.words('english'))
        # Porter stemmer instance for stemming words
        self.stemmer = PorterStemmer()
        # Dictionary to store computed tf-idf values
        self.tfidf_dict = {}

    def tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenizes, lowers and stems the input text."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenization
        words = word_tokenize(text)
        # Remove stopwords and apply stemming
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return words

    def compute_tf_idf(self, texts: List[str], max_df: float = 0.80, min_df: float = 0.16) -> np.array:
        """Computes the TF-IDF for the given list of texts."""
        num_documents = len(texts)

        # Compute term frequency
        tf = {}
        doc_freq = {}  # This will help in calculating the document frequency for each term

        for idx, text in enumerate(texts):
            tokens = self.tokenize_and_stem(text)
            tf[idx] = {}

            unique_tokens = set(tokens)  # To ensure we count each term once for each document

            for token in tokens:
                tf[idx][token] = tf[idx].get(token, 0) + 1

            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Filter terms based on max_df and min_df
        # terms_to_remove = {term for term, freq in doc_freq.items() if
        #                    freq / num_documents > max_df or freq / num_documents < min_df}
        terms_to_remove = {
            term  # Add the term to the set
            for term, freq in doc_freq.items()  # For each term and its document frequency
            if (
                (print(f"Removing {term} (appears in {freq / num_documents:.2%} of documents) - too common") or True)
                if freq / num_documents > max_df  # If the term is too common (appears in more than max_df of documents)
                else (
                    (print(f"Removing {term} (appears in {freq / num_documents:.2%} of documents) - too rare") or True)
                    if freq / num_documents < min_df  # Or if the term is too rare (appears in less than min_df of documents)
                    else False
                )
            )  # If either of the above conditions is true, the term is added to the set
        }
        print(f'Number of terms to remove: {len(terms_to_remove)}')
        print(f'max_df: {max_df}, min_df: {min_df}')

        for idx in tf:
            for term in terms_to_remove:
                tf[idx].pop(term, None)

        # Compute inverse document frequency
        idf = {}
        for token in doc_freq:
            if token not in terms_to_remove:
                idf[token] = np.log(num_documents / doc_freq[token])

        # Compute TF-IDF
        for idx in tf:
            self.tfidf_dict[idx] = {}
            for token in tf[idx]:
                self.tfidf_dict[idx][token] = tf[idx][token] * idf[token]

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Computes the cosine similarity between two vectors represented as dictionaries."""
        dot_product = sum([vec1.get(token, 0) * vec2.get(token, 0) for token in set(vec1.keys()).union(vec2.keys())])
        mag1 = np.sqrt(sum([vec1.get(token, 0)**2 for token in vec1]))
        mag2 = np.sqrt(sum([vec2.get(token, 0)**2 for token in vec2]))
        if not mag1 or not mag2:
            return 0.0
        return dot_product / (mag1 * mag2)

    def compute_similarity(self, fact, passages):
        """Computes the similarity between a fact and a list of passages."""
        texts = [fact] + [p['text'] for p in passages]
        self.compute_tf_idf(texts)
        # Compute the cosine similarity between the fact and each passage
        similarities = [self.cosine_similarity(self.tfidf_dict[0], self.tfidf_dict[i+1]) for i in range(len(passages))]
        return similarities

    def predict(self, fact: str, passages: List[dict]) -> str:
        # raise Exception("Implement me")
        """Predicts whether the fact is supported by any of the given passages."""
        similarities = self.compute_similarity(fact, passages)
        # Decision based on the max similarity with any passage
        if np.max(similarities) > self.threshold:
            return "S"  # Supported
        else:
            return "NS"  # Not Supported

class EntailmentFactChecker(object):
    def __init__(self, ent_model, threshold=0.5):
        self.ent_model = ent_model
        self.threshold = threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        # raise Exception("Implement me")
        """Predicts whether the fact is supported by any of the given passages."""
        max_entailment_score = 0

        for passage in passages:
            sentences = nltk.sent_tokenize(passage['text'])
            for sentence in sentences:
                entailment_score = self.ent_model.check_entailment(sentence, fact)
                max_entailment_score = max(max_entailment_score, entailment_score)

        if max_entailment_score > self.threshold:
            return "S"
        else:
            return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

