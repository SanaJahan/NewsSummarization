# News Classification and Summarization Services

#### We created two UIs for our service. 
Swagger
First, we created a Swagger API interface with both our services (summarization and classification). This can be found in the following files:

```news_api.py```

```news_api.yaml```

This user interface has the class Swagger interface, and also has health checks for both the classification and summarization services. In order to run the Swagger API, please run the following commands from the APIs folder in your terminal: 
```python news_api.py```

Then, in your browser, navigate to:

```http://localhost:8080/ui/```

Please note that the Swagger interface is not dockerized.

The user interacts with this interface the same way they would with any swagger interface: the user would select the API they want to use, i.e. a health check, classification, or summarization, and then pass the parameters that are described in the API, and execute. Below is a screenshot from our Swagger interface with example instructions for the classification API.

Figure 11: News Classification and Summarization Swagger Interface
![Figure 11](https://github.com/SanaJahan/NewsSummarization/blob/main/Project%20Report.png)

Streamlit
For the more interactive UI (and our primary UI), we created a streamlit application. This application is dockerized, and can either be run with the docker file, or simply as a streamlit application. We created this UI since it allows for a much cleaner and user friendly interface for the user when using our application. Instead of describing how the user would use this in words, we have created a video demo of the streamlit app running in a Docker container below.

In order to run the streamlit app on your local device, as mentioned previously, this can be done by just running the streamlit app directly in your terminal with the following command:

```streamlit run streamlit-news-app.py```


However, an easier way to do this (and avoid errors due to package dependencies) would be to run this using Docker. The Dockerfile is included in the APIs folder of our github. After cloning this repository, and assuming Docker is installed on your local machine, you may run the following commands:

```docker build . -t news-app```

```docker run -d -p 8501:8501 --name news-app-docker news-app:latest```

Then, in your browser, navigate to:

```http://0.0.0.0:8501/```

The News Classification and Summarization App should be running in your browser at the above address.

Please find the video demo of our streamlit app at the following YouTube link:
https://youtu.be/--k8BTIsXSo 

