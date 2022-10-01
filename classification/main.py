import os
import warnings
import numpy as np
from datetime import datetime
import nltk
import re
import json
from google.cloud import vision
import traceback
from joblib import Parallel, delayed
from google.cloud import storage
from sklearn.pipeline import Pipeline
from joblib import dump, load
import pandas as pd
import functions_framework

pd.set_option('max_colwidth', 500)
pd.set_option('display.max_rows', 500)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
warnings.filterwarnings("ignore")

stopwords = set(nltk.corpus.stopwords.words())
lemma = nltk.WordNetLemmatizer()


def async_detect_document(gcs_source_uri, gcs_destination_uri):
    """OCR with PDF/TIFF as source files on GCS"""
    # Supported mime_types are: 'application/pdf' and 'image/tiff'
    mime_type = 'application/pdf'

    # How many pages should be grouped into each json output file.
    batch_size = 2
    print(gcs_source_uri, '\n', gcs_destination_uri)
    client = vision.ImageAnnotatorClient()

    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source,
                                      mime_type=mime_type)

    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(gcs_destination=gcs_destination,
                                        batch_size=batch_size)

    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature],
        input_config=input_config,
        output_config=output_config)

    operation = client.async_batch_annotate_files(requests=[async_request])

    print('Waiting for the operation to finish.')
    operation.result(timeout=420)

    # Once the request has completed and the output has been
    # written to GCS, we can list all the output files.
    storage_client = storage.Client()
    match = re.match(r'gs://([^/]+)/(.+)', gcs_destination_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)

    bucket = storage_client.get_bucket(bucket_name)

    # List objects with the given prefix, filtering out folders.
    blob_list = [
        blob for blob in list(bucket.list_blobs(prefix=prefix))
        if not blob.name.endswith('/')
    ]
    print('Output files:')
    text = []
    for blob in blob_list:
        name = blob.name
        json_string = blob.download_as_string()
        data = json.loads(json_string)
        for pages in data['responses']:
            try:
                text.append(pages['fullTextAnnotation']['text'].replace(
                    '-\n', '').replace('\n', ' '))
            except:
                print(traceback.format_exc())
                text.append('{}'.format(np.nan))
                print(name)
        # all_text = [{v+1: k for v, k in enumerate(text)}]
        all_text = " ".join(text)
        blob.delete()
    return all_text


def remove_url(text):
    text = text.replace('-\n', '')
    text = text.encode().decode()
    text = text.replace('\n', ' ')
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def clean_stopwords(text, stopwords):
    res = []
    for word in text:
        if word not in stopwords:
            res.append(word)
    return res


def lemmatize(tokens_no_sw, lemma):
    tokens_no_sw_lemma = [lemma.lemmatize(each) for each in tokens_no_sw]
    return tokens_no_sw_lemma


def bigram(text):
    return list(nltk.bigrams(text))


def cleaning(text):
    text = text.replace('.', '')
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    text = re.sub('(?<=\d)[,.](?=\d)', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('[!@#$]', '', text)
    text = re.sub('[^a-z \n\.]', '', text)

    return text


def get_root(bucket):
    # Get the root directory for the bucket.
    lst = []
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket)
    # Loop over blobs in the bucket and print out some properties.
    print("Blobs:")
    for blob in blobs:
        if re.match('^[0-9][0-9]', str(blob.name.split("/")[0])):
            lst.append(blob.name.split("/")[0])
    root = max(lst, key=lambda d: datetime.strptime(d, '%d-%m-%Y'))
    return root


def func_blob(blob, bucket_uri, root, destination_folder):
    try: # if the file is not a pdf or tiff
        if '.pdf' in blob.name:
            file_path = blob.name
            # print(file_path)
            gcs_source_uri = bucket_uri + file_path
            gcs_destination_uri = bucket_uri + destination_folder + file_path.split(
                '.')[0] + '/'
            all_text = async_detect_document(gcs_source_uri,
                                             gcs_destination_uri)
            response = {'name': [file_path], 'text': all_text}
            return pd.DataFrame(response)
        else:
            response = {'name': [np.nan], 'text': [np.nan]}
            return pd.DataFrame(response)
    except Exception as e:
        print(e)
        pass


def main():
    bucket = "crvl-dev-etmf-downloaded-pdfs"
    bucket_uri = 'gs://crvl-dev-etmf-downloaded-pdfs/'
    root = get_root(bucket)
    destination_folder = 'vision_ocr_json/'
    all_files = []
    df = pd.DataFrame(columns=['name', 'text'])
    # Instantiates a client
    client = storage.Client()
    # Get GCS bucket
    bucket = client.get_bucket(bucket)
    # Get blobs in bucket (including all subdirectories) and add specific subirectory
    blobs = bucket.list_blobs(prefix=root)
    #select only pdf files
    all_data = Parallel(n_jobs=-1, prefer="threads")(
        delayed(func_blob)(blob, bucket_uri, root, destination_folder)
        for blob in blobs)
    for item in all_data:
        df = pd.concat([df, item], axis=0)
    data = df.dropna().reset_index(drop=True)
    return data


def get_file(path):
    # get the file from the path
    bucket_name = 'cerevel_trip_repository'
    # Initialise a client
    storage_client = storage.Client()
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the filepath
    blob = bucket.blob(path)
    name = path.split("/")[2]
    # Download the file to a destination
    blob.download_to_filename(f'/tmp/{name}')


def get_file2(path):
    try:
        bucket_name = 'cerevel_trip_repository'
        # Initialise a client
        storage_client = storage.Client()
        # Create a bucket object for our bucket
        bucket = storage_client.get_bucket(bucket_name)
        # Create a blob object from the filepath
        blob = bucket.blob(path)
        name = path.split("/")[1]
        # Download the file to a destination
        if "pkl" in name:
            blob.download_to_filename(f'/tmp/{name}')
    except:
        exit


def get_path():
    bucket = "cerevel_trip_repository"
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket)
    lst2 = []
    print("Blobs:")
    for blob in blobs:
        if "classification_models" in blob.name:
            lst2.append(blob.name)
    return lst2


def download_model():
    # method to download the model from GCS
    lst2 = []
    path3 = get_path()
    for path in path3:
        try:
            get_file(path)
            print(path)
        except:
            print(path)
            get_file2(path)
            continue


def move_blob(bucket_name, blob_name, destination_bucket_name,
              destination_blob_name):
    """Moves a blob from one bucket to another with a new name."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of your GCS object
    # blob_name = "your-object-name"
    # The ID of the bucket to move the object to
    # destination_bucket_name = "destination-bucket-name"
    # The ID of your new GCS object (optional)
    # destination_blob_name = "destination-object-name"

    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    blob_copy = source_bucket.copy_blob(source_blob, destination_bucket,
                                        destination_blob_name)
    source_bucket.delete_blob(blob_name)

    print("Blob {} in bucket {} moved to blob {} in bucket {}.".format(
        source_blob.name,
        source_bucket.name,
        blob_copy.name,
        destination_bucket.name,
    ))


@functions_framework.http
def pdf_classification(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    df = main()
    download_model()
    tf_idf = load('/tmp/tf_idf.pkl')
    lr = load('/tmp/lr_clasfctn_model_1.pkl')
    tf_idf_i = load('/tmp/tf_idf_i.pkl')
    lr_i = load('/tmp/lr_clasfctn_model_i.pkl')
    tf_idf_s = load('/tmp/tf_idf_s.pkl')
    lr_s = load('/tmp/lr_clasfctn_model_s.pkl')
    
    # pipeline is the pipeline to run the classification
    pipe = Pipeline([('vectorization', tf_idf), ('prediction', lr)])
    
    pipe_syn = Pipeline([('vectorization', tf_idf_s), ('prediction', lr_s)])
    pipe_iqv = Pipeline([('vectorization', tf_idf_i), ('prediction', lr_i)])

    lst = pipe.predict(df.text)
    list_string = map(str, lst)
    lst = list(list_string)
    lst = list(
        map(lambda x: x.replace('1.0', 'iqvia').replace('0', 'syneos'), lst))

    df['doc_type'] = lst
    df["doc_type2"] = " "

    for i in range(len(df)):
        if df.doc_type[i] == 'iqvia':
            lst = pipe_iqv.predict([df.text[i]])
            list_string = map(str, lst)
            lst = list(list_string)
            lst = list(
                map(
                    lambda x: x.replace('1', 'iqvia_remote_interim').replace(
                        '0', 'iqvia_site_visit'), lst))
            df["doc_type2"][i] = lst[0]
        if df.doc_type[i] == 'syneos':
            lst = pipe_syn.predict([df.text[i]])
            list_string = map(str, lst)
            lst = list(list_string)
            lst = list(
                map(
                    lambda x: x.replace('1', 'syneos_interim').replace(
                        '0', 'syneos_site_visit'), lst))
            df["doc_type2"][i] = lst[0]

    client = storage.Client()
    # bucket_iqvia = 'crvl-test-ml-classification'
    # bucket_iqvia2 = 'crvl-test-ml-classification'
    # bucket_syn = 'crvl-test-ml-classification/SYNEOS_TEMP1'
    # bucket_syn2 = 'crvl-test-ml-classification/SYNEOS_TEMP2'
    source = 'crvl-dev-etmf-downloaded-pdfs'
    destination = 'crvl-test-ml-classification'

    src_dest = {}
    for i in range(len(df)):
        if df.doc_type2[i] == 'iqvia_remote_interim' or df.doc_type2[
                i] == 'iqvia_site_visit':
            move_blob(source, df.name[i], destination,
                      f'IQVIA_TEMP1/{df.name[i]}')
            src_dest[df.doc_type2[i]] = {
                'source': source,
                'source_blob_name': df.name[i],
                'destination': f'IQVIA_TEMP1/{df.name[i]}'
            }
        if df.doc_type2[i] == 'syneos_interim':
            move_blob(source, df.name[i], destination,
                      f'SYNEOS_TEMP1/{df.name[i]}')
            src_dest[df.doc_type2[i]] = {
                'source': source,
                'source_blob_name': df.name[i],
                'destination': f'SYNEOS_TEMP1/{df.name[i]}'
            }
        if df.doc_type2[i] == 'syneos_site_visit':
            move_blob(source, df.name[i], destination,
                      f'/SYNEOS_TEMP2/{df.name[i]}')
            src_dest[df.doc_type2[i]] = {
                'source': source,
                'source_blob_name': df.name[i],
                'destination': f'SYNEOS_TEMP2/{df.name[i]}'
            }

    os.remove('/tmp/tf_idf.pkl')
    os.remove('/tmp/lr_clasfctn_model_1.pkl')
    os.remove('/tmp/tf_idf_i.pkl')
    os.remove('/tmp/lr_clasfctn_model_i.pkl')
    os.remove('/tmp/tf_idf_s.pkl')
    os.remove('/tmp/lr_clasfctn_model_s.pkl')
    print('All file moved to the destination bucket')
    iqvia_URL = ''
    call_entity_extraction(iqvia_URL, iqvia_URL)
    syneos_URL = 'https://us-east1-trip-report-344816.cloudfunctions.net/crvl-dev-cf-syneos-ee'
    call_entity_extraction(syneos_URL, syneos_URL)
    return src_dest


def call_entity_extraction(endpoint, audience):
    """
    make_authorized_get_request makes a GET request to the specified HTTP endpoint
    by authenticating with the ID token obtained from the google-auth client library
    using the specified audience value.
    """
    # Cloud Functions uses your function's URL as the `audience` value
    # audience = https://project-region-projectid.cloudfunctions.net/myFunction
    # For Cloud Functions, `endpoint` and `audience` should be equal

    req = urllib.request.Request(endpoint)

    auth_req = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)

    req.add_header("Authorization", f"Bearer {id_token}")
    response = urllib.request.urlopen(req)

    return response.read()