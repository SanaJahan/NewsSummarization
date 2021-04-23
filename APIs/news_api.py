"""
To run this app, in your terminal:
> python news_api.py

Navigate to:
> http://localhost:8080/ui/
"""
from time import sleep
import connexion
import pandas as pd
import nltk
import os
import glob
import re
from newspaper import Article
from newspaper.article import ArticleDownloadState, ArticleException
from nltk.tokenize import sent_tokenize
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import numpy as np
from rouge_metric import PyRouge
import matplotlib.pyplot as plt
from joblib import load
import pickle

nltk.download('punkt')  # one time execution
nltk.download('stopwords')  # one time execution

stop_words = stopwords.words('english')

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='')
application = app.app

clf = load('Classification.joblib')
# load our pre-trained vectorizer
tfidf = pickle.load(open("tfidf.pickle", "rb"))

recall_scores_list = []
f_scores_list = []
fuzz_ratio = []

# Dictionary with all our mapped categories
category_dict = {
    0: "World News",
    1: "Media",
    2: "Black Voices",
    3: "Entertainment",
    4: "Crime",
    5: "Comedy",
    6: "Politics",
    7: "Women",
    8: "Queer Voices",
    9: "Latino Voices",
    10: "Religion",
    11: "Education",
    12: "Science",
    13: "Tech",
    14: "Business",
    15: "Sport",
    16: "Travel",
    17: "Impact"
}


# Implement a simple health check function (GET)
def health_summary():
    # Test to make sure our service is actually healthy
    try:
        summarize("https://www.huffingtonpost.com/entry/hugh-grant-marries_us_5b09212ce4b0568a880b9a8c", 50)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}


# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        features = tfidf.transform(['Hello I am a sports article!'])
        clf.predict(features)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}


word_embeddings = {}


class ExtractiveTextSummarizer:

    def __init__(self):
        # Extract word vectors
        f = open("glove.6B.50d.txt", 'r', errors='ignore', encoding='utf8')
        # f = open('../model/glove.6B.50d.txt', encoding='windows-1252')
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except StopIteration:
                f.__next__()
            except:
                pass
            word_embeddings[word] = coefs
        f.close()

    # text cleaning
    def preprocessing(self, sentences):
        # remove punctuations, numbers and special characters
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        clean_sentences = [s.lower() for s in clean_sentences]
        return clean_sentences

    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    def create_sentence_vectors(self, clean_sentences):
        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((50,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((50,))
            sentence_vectors.append(v)
        return sentence_vectors

    # find similarity between sentences using cosine-similarity
    def create_similarity_matrix(self, sentences, sentence_vectors):
        sim_mat = np.zeros([len(sentences), len(sentences)])
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = \
                        cosine_similarity(sentence_vectors[i].reshape(1, 50), sentence_vectors[j].reshape(1, 50))[0, 0]

        return sim_mat

    # convert similarity matrix into a graph using page rank algorithm.
    # Nodes of the graph will be sentences and edges will be similarity scores
    # extract the top N scored sentences
    def page_rank(self, sim_mat, sentences, summary_length):
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        final_sentences = ''
        for i in range(summary_length):
            final_sentences += ''.join(ranked_sentences[i][1])
        return final_sentences

    def create_summary(self, article, summary_length):
        sentences = [sent_tokenize(article)]
        # title = []
        # for line in open(article, 'r'):
        #     json_entry = (json.loads(line))
        #     if line != '':
        # break the sentences into individual sentences
        # title.append(json_entry['title'])
        # flatten the list
        sentences = [y for x in sentences for y in x]
        summary_length = int((summary_length / 100) * len(sentences))
        clean_sentences = self.preprocessing(sentences)
        sentences_vector = self.create_sentence_vectors(clean_sentences)
        sim_matrix = self.create_similarity_matrix(sentences, sentences_vector)
        summary = self.page_rank(sim_matrix, sentences, summary_length)

        return summary

    def casefolding(self, sentence):
        return sentence.lower()

    def cleaning(self, sentence):
        return re.sub(r'[^a-z]', ' ', re.sub("’", '', sentence))

    def tokenization(self, sentence):
        return sentence.split()

    def sentence_split(self, paragraph):
        return nltk.sent_tokenize(paragraph)

    def word_freq(self, data):
        w = []
        for sentence in data:
            for words in sentence:
                w.append(words)
        bag = list(set(w))
        res = {}
        for word in bag:
            res[word] = w.count(word)
        return res

    def summary_ranking(self, news_text, n):
        sentence_list = self.sentence_split(str(news_text))
        data = []
        for sentence in sentence_list:
            data.append(self.tokenization(self.cleaning(self.casefolding(sentence))))
        data = (list(filter(None, data)))
        wordfreq = self.word_freq(data)
        ranking = []
        for words in data:
            temp = 0
            for word in words:
                temp += wordfreq[word]
            ranking.append(temp)

        result = ''
        sort_list = np.argsort(ranking)[::-1][:n]
        # print(sort_list)
        # l = []
        for i in range(n):
            result += '{} '.format(sentence_list[sort_list[i]])
        return result


def summarize(url, summary_length):
    """
        :param summary_length: length of the summary in percentage
        :param url: url to scrape from the web
        :return: call the summarize function
        """
    # url = url.strip("https://")
    # print(url)
    article_huff = Article(url)
    slept = 0
    article_huff.download()
    while article_huff.download_state == ArticleDownloadState.NOT_STARTED:
        # Raise exception if article download state does not change after 12 seconds
        if slept > 13:
            raise ArticleException('Download never started')
    sleep(1)
    slept += 1

    article_huff.parse()
    summarizer = ExtractiveTextSummarizer()
    summary = summarizer.create_summary(article_huff.text, summary_length)
    return {"title": article_huff.title, "summary": summary}


def wf_summarize(url, summary_length):
    """
        :param summary_length: length of the summary in percentage
        :param url: url to scrape from the web
        :return: call the summarize function
    """
    article_huff = Article(url)
    slept = 0
    article_huff.download()
    while article_huff.download_state == ArticleDownloadState.NOT_STARTED:
        # Raise exception if article download state does not change after 12 seconds
        if slept > 13:
            raise ArticleException('Download never started')
    sleep(1)
    slept += 1
    n = int(summary_length)

    article_huff.parse()
    news_text = article_huff.text
    summarizer = ExtractiveTextSummarizer()
    summary = summarizer.summary_ranking(news_text, n)
    return {"title": article_huff.title, "summary": summary}


def evaluation_metrics(summaries, hypotheses_list):
    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                    rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
    actual_summary_list = []
    references_list = []
    folder_path = "../BBCNewsSummary/Summaries/business"
    for filename in glob.glob(os.path.join(folder_path, '*.txt')):
        with open(filename, 'r') as f:
            text = f.read()
            references = []
            actual_summary_list.append(text)
            # Pre-process and tokenize the summaries as you like
            references.append(text.split())
            references_list.append(references)

    for i in range(len(summaries)):
        fuzz_ratio.append(fuzz.ratio(summaries[i], actual_summary_list[i]))
        scores = rouge.evaluate_tokenized(hypotheses_list[i], references_list[i])
        recall_scores_list.append(scores['rouge-1']['r'])
        f_scores_list.append(scores['rouge-1']['f'])
    # return fuzz_ratio, recall_scores_list, f_scores_list


def fuzzy_visualize():
    plt.hist(fuzz_ratio, bins=len(fuzz_ratio))
    plt.xlabel('Levenshtein distance score')
    plt.show()


def recall_visualize():
    plt.hist(recall_scores_list, density=True, bins=len(recall_scores_list))
    plt.ylabel('Rouge recall score')
    plt.show()


def fscore_visualize():
    plt.hist(f_scores_list, density=True, bins=len(f_scores_list))
    plt.ylabel('F-Score')
    plt.xlabel('Data')
    plt.show()


def generate_summary(summary_length):
    summary_list = []
    hypotheses_list = []

    j = 0
    folder_path = "../BBCNewsSummary/NewsArticles/business"
    for filename in glob.glob(os.path.join(folder_path, '*.txt')):
        with open(filename, 'r') as f:
            j = j + 1
            text = f.read()
            summarizer = ExtractiveTextSummarizer()
            # change method names to call different algorithms here
            summary = summarizer.summary_ranking(text, summary_length)
            summary_list.append(summary)
            # Pre-process and tokenize the summaries as you like
            hypotheses = [text.split()]
            hypotheses_list.append(hypotheses)
            if j == summary_length:
                break
    evaluation_metrics(summary_list, hypotheses_list)


# Implement our predict function
def predict(article):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()

    features = tfidf.transform([article])
    prediction = clf.predict(features)
    prediction = category_dict.get(prediction[0])

    # Return the prediction as a json
    return {"prediction": prediction}


# Read the API definition for our service from the yaml file
app.add_api("news_api.yaml")

# # Start the app
if __name__ == "__main__":
    app.run()

# VISUALIZATION FUNCTIONS
# generate_summary(8)
# fuzzy_visualize()
# fscore_visualize()
# # recall_visualize()
# summarize("https://www.huffingtonpost.com/entry/hugh-grant-marries_us_5b09212ce4b0568a880b9a8c", 5)
