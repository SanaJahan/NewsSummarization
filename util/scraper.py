import json
from time import sleep
import io
from newspaper import Article
from newspaper.article import ArticleException, ArticleDownloadState


def read_file(path):
    """
    :param path: the path of the file containing the urls
    :return: list of the urls to be scraped
    """
    url_list = []
    for line in open(path, 'r'):
        json_entry = (json.loads(line))
        url_list.append(json_entry['link'])

    return url_list


def scrape_url(url_list):
    """
    :param url_list: list of urls to scrape from the web
    :return: nothing, just create a text file containing article text of each url
    """

    for i in range(0, len(url_list)):
        article_huff = Article(url_list[i])  # iterate through each article link
        slept = 0
        article_huff.download()
        while article_huff.download_state == ArticleDownloadState.NOT_STARTED:
            # Raise exception if article download state does not change after 11 seconds
            if slept > 12:
                raise ArticleException('Download never started')
        sleep(1)
        slept += 1

        article_huff.parse()
        article_info_huff = article_huff.text
        file_name = "../huffpostarticles/huffpost" + str(i) + ".txt"
        with io.open(file_name, "w", encoding="utf-8") as f:
            f.write(article_info_huff)


def __main__():
    urls = read_file("../huffpost/News_Category_Dataset_v2.json")
    scrape_url(urls)


__main__()


# We can write each of the following properties if required for classification or summarization
# article_info_huff = {'title': article_huff.title,
#                      'description': article_huff.meta_description,
#                      'text': article_huff.text,
#                      'publisher': 'huff'}
