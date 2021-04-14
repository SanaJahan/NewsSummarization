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
    category_list = []
    for line in open(path, 'r'):
        json_entry = (json.loads(line))
        url_list.append(json_entry['link'])
        category_list.append(json_entry['category'])

    return category_list, url_list


def scrape_url(category_list, url_list):
    """
    :param url_list: list of urls to scrape from the web
    :return: nothing, just create a text file containing article text of each url
    """

    for i in range(0, len(url_list)):
        article_huff = Article(url_list[i])  # iterate through each article link
        slept = 0
        article_huff.download()
        while article_huff.download_state == ArticleDownloadState.NOT_STARTED:
            # Raise exception if article download state does not change after 12 seconds
            if slept > 13:
                raise ArticleException('Download never started')
        sleep(1)
        slept += 1

        article_huff.parse()
        article_info_huff = {"category": category_list[i], "title": article_huff.title, "text": article_huff.text}
        file_name = "../huffpostarticles/huffpost" + str(i) + ".json"
        with io.open(file_name, "w", encoding="utf-8") as f:
            f.write(json.dumps(article_info_huff))


def __main__():
    category_list, urls = read_file("../huffpost/News_Category_Dataset_v2.json")
    scrape_url(category_list, urls)


__main__()


# We can write each of the following properties if required for classification or summarization
# article_info_huff = {'title': article_huff.title,
#                      'description': article_huff.meta_description,
#                      'text': article_huff.text,
#                      'publisher': 'huff'}
