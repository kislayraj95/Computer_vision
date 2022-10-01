# Program Description

This program contains the full stack implementation of the cloud architecture given in the paper.

## Summary
The entire workflow of the program has been explained as we go through the documentation. The pdfs are extracted and obtained from the server and then classified using the trained model stored in the server, in the server and the final and well classified information is generated which is fetched into the local storage.

# Component Specifications

* eTMF Server: The electronic Trial Master File (eTMF) server software and server technology to guide and assist the setup, collection, storage, tracking and archival of essential clinical study documents. The eTMF server is a web application that is hosted on Google Cloud Platform. It provides a secure, scalable, and reliable solution for storing and managing clinical trial documents.

* Cloud Scheduler: It allows you to schedule virtually any job, including batch, big data jobs, cloud 
infrastructure operations, and more.

* Cloud Functions to integrate with eTMF

* Cloud Functions to classify PDFs into specific types

* Cloud Storage buckets for specific document types

* Cloud Storage for storing classification models

* API to fetch the classification results

* Authentication for the API


# Main Workflow

## Fetching and preprocessing the data from the server

* The emtp server contains a bunch of PDF files which have been classified into different types. 

* we use the cloud functions to integrate with the eTMF server and fetch the PDF from the cloud scheduler. Initially the root of the bucket URI is set to the cloud storage bucket.

* The get_root() method is used to fetch the filtered root of the bucket of the blobs listed by the storage client. Only those blobs which have name in the correct format as filtered by the regex pattern named upto 2 digits from 0 to 9 are stored and rest are ignored.


* A client is instantianted and the Google cloud services bucket is obtained in the variable called 'bucket'. Now all the blobs in the bucket are fetched including the subdirectories.
 
 
* Cloud scheduler: The blobs in the bucket are classied into various types from which only PDFs are selected. This is an asynchronous process hence multithreaded traversal is used to fetch and filter the correct blobs and then concantenate the dataframe with their names and the text they contain.

* Cleaning steps are performed on the dataframe to remove the unwanted datatypes and miscallaneous text as follows:
    1. Detect the language of the text
    2. Remove the urls
    3. Remove the special characters
    4. Remove HTML tags
    5. Remove emoji characters
    6. Remove the empty rows, null values and NaN values, special characters
    7. Remove the stopwords
    8. Remove the punctuation
    9. Lemmatize the words
    10. Biggrams are generated (generate such word pairs from the existing sentence maintain their current sequences)


    The above process returns a well formatted dataframe which goes into the cloud storage.

## Document Extraction and Pdf Classification

* After obtaining the dataframe, the dataframe is passed to the pdf_classifiication() method through the data ingestion pipeline over an HTTP request which gets triggered each time. The classifier is trained on the dataframe and then the classifier is used to classify the PDFs. The classification is done using the sklearn library.

* The classification model is downloaded if it is not already downloaded within the cloud. There are several bearer tokens and authentication keys which are locally catched to the temp directory. The bearer token is used to authenticate the requests to the API.

* The Cloud Vision API is used to extract the text from the PDFs. The text is then passed to the classifier to classify the PDFs. Cloud storage buckets for the classification models are created. The classification model is stored in the cloud storage bucket.

* The sklearn.pipeline.Pipeline combines the classifier and the vectorizer. The vectorizer is used to convert the text into a vector. The classifier is used to classify the vector. The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. For this, it enables setting parameters of the various steps using their names and the parameter name separated by a ‘__’.

* It was found that there can be two types of documents i.e. 'doc_type1' and 'doc_type2'hence the classifier is trained on two different dataframes. The first dataframe is the one which contains the PDFs which are classified. there are four bucjets for the classification models. The first bucket is for the classifier trained on the PDFs which are classified. The second bucket is for the classifier trained on the PDFs which are not classified. The third bucket is for the classifier trained on the PDFs which are classified and the fourth bucket is for the classifier trained on the PDFs which are not classified. These are bucket_iqvia, bucket_iqvia2, bucket_syn and bucket_syn2.

* The last step is do entity extration to make an authorized GET request to the specified HTTP endpoint by authenticating with the ID token obtained from the google-auth client library using the specified audience value. 

The temporary cache files are removed after the classification is done. Now the data is ready for packing and hence and all the files are then moved to the destination bucket in the user directory.


## Final output

The final output are zip files (packages) named as the following:
* wordnet.zip: Adjectives, adverbs, verbs, nouns lexnames etc.

* stopwords.zip: All various languages stopwords

* omw*.zip: Citations of the differnt worldwide languages

All the zip files are extracted and the contents are stored in the respective folders ascynchronously.

The tokenizer folder (one level above) contains various bearer and auth tokens which were used to fetch the data from the server after authentication.