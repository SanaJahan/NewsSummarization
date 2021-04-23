FROM continuumio/miniconda3
RUN pip3 install nltk
RUN pip3 install newspaper3k
RUN pip3 install networkx
RUN pip3 install sklearn
RUN pip3 install fuzzywuzzy
RUN pip3 install numpy
RUN pip3 install pyrouge
RUN pip3 install matplotlib
RUN pip3 install joblib
RUN pip3 install pandas
RUN pip3 install pyrouge
RUN pip3 install rouge_metric

#Install python packages
RUN mkdir /docker-folder
COPY requirements.txt /docker-folder/
RUN pip install -r /docker-folder/requirements.txt

#Copy files into container
COPY Classification.joblib /docker-folder/
COPY tfidf.pickle /docker-folder/
COPY glove.6B.50d.txt /docker-folder/
COPY streamlit-news-app.py /docker-folder/
COPY pyrouge /docker-folder/

#Set work directory and open the required port
WORKDIR /docker-folder
EXPOSE 8501

# Run our service script
CMD ["streamlit", "run", "streamlit-news-app.py"]