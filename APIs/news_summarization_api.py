"""
To run this app, in your terminal:
> python news_classification_api.py

Navigate to:
> http://localhost:8080/ui/
"""
from time import sleep
import connexion
import numpy as np
import pandas as pd
import nltk
from newspaper import Article
from newspaper.article import ArticleDownloadState, ArticleException
from nltk.tokenize import sent_tokenize
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords

nltk.download('punkt')  # one time execution
nltk.download('stopwords')  # one time execution

stop_words = stopwords.words('english')

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='')
application = app.app


# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        summarize("https://www.huffingtonpost.com/entry/hugh-grant-marries_us_5b09212ce4b0568a880b9a8c", 50)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}


word_embeddings = {}


class ExtractiveTextSummarizer:

    def __init__(self):
        # Extract word vectors
        f = open("../glove.6B.50d.txt", 'r', errors='ignore', encoding='utf8')
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
    # Nodes of the graph will be sentences and endges will be similarity scores
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


# Implement our predict function
# def summarize(article, title, summary_length):
#     summarizer = ExtractiveTextSummarizer()
#     summary = summarizer.create_summary(article, summary_length)
#     # print(title)
#     # print(summary)
#     return {"title": title, "summary": summary}


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
    # print(article_huff.title)
    # print(summary)
    return {"title": article_huff.title, "summary": summary}


# Read the API definition for our service from the yaml file
app.add_api("news_summarization_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()

# summarize("https://www.huffingtonpost.com/entry/hugh-grant-marries_us_5b09212ce4b0568a880b9a8c", 50)
