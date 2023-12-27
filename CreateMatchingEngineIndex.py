#CREATE MATCHING ENGINE INDEX FROM pdf


from google.cloud import aiplatform,storage,bigquery
import os,PyPDF2
import pandas as pd

PROJECT_ID = "gcp-project-0523"
REGION     = 'us-central1'
BUCKET     = 'gcp-project-0523-ann-bucket'
BUCKET_URI = 'gs://gcp-project-0523-ann-bucket'

aiplatform.init(project        = PROJECT_ID,
                location       = REGION,
                staging_bucket = BUCKET_URI)

#READ pdf DOCUMENT
def read_pdf_document(pdf_name):
    #reader = PyPDF2.PdfReader('mlops_for_dummies_databricks_special_edition.pdf')
    reader = PyPDF2.PdfReader(pdf_name)
    pg_cnt = len(reader.pages)
    print('The number of pages are : ',str(pg_cnt))

    book_txt_lst=[]
    for n in range(8,pg_cnt):
        txt=''
        txt = reader.pages[n].extract_text().replace('\t',' ').replace('\n',' ')
        txt_lst = txt.split('.')
        for item in txt_lst:
            book_txt_lst.append(item)


    #REMOVE DUPLICATES FROM THE LIST
    book_txt_lst_clean = list(set(book_txt_lst))

    #CREATE FINAL LIST OF pdf DOCUMENT SENTENCES
    book_txt_lst_final=[]
    idx=0
    for item in book_txt_lst_clean:
        item.strip()
        t=(idx,item)
        book_txt_lst_final.append(t)
        idx+=1
    return book_txt_lst_final

    
#CREATE DATAFRAME
df = pd.DataFrame(book_txt_lst_final,columns=['ids','Text'])



#GET ENCODER FROM TENSORFLOW HUB
from tensorflow_encoder import get_encoder

encoder = get_encoder()


#CREATE EMBEDDINGS

from text_embeddings import encode_text_to_embedding
import json
#questions = df.Text.tolist()[1:607]
#question_embeddings = encode_text_to_embedding(text_encoder=encoder, sentences=questions)

BATCH_SIZE = 100
questions = df.Text.tolist()[:607]
embeddings_file_name='embeddings_file3.json'
print('# of questions are : ', str(len(questions)))

with open(embeddings_file_name, "w") as f:
    for i in tqdm(range(0, len(questions), BATCH_SIZE)):
        id_chunk = ids[i : i + BATCH_SIZE]
        question_chunk_embeddings = encode_text_to_embedding(text_encoder=encoder,
                                                             sentences=questions[i : i + BATCH_SIZE])
        # Append to file
        embeddings_formatted = [
            json.dumps(
                {
                    "id": str(id),
                    "embedding": [str(value) for value in embedding],
                }
            )
            + "\n"
            for id, embedding in zip(id_chunk, question_chunk_embeddings)
        ]
        f.writelines(embeddings_formatted)
        print(i)


#COPY EMBEDDINGS TO GCS BUCKET
UNIQUE_FOLDER_NAME = "pdf_st5_large_embeddings"
remote_folder = f"{BUCKET_URI}/{UNIQUE_FOLDER_NAME}/"
! gsutil cp {embeddings_file_name} {remote_folder}


#CREATE MATCHING ENGINE INDEX

DISPLAY_NAME = "pdf_st5_large_embeddings_index"
DESCRIPTION  = "MLOps pdf document"

tree_ah_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(display_name                = DISPLAY_NAME,
                                                                    contents_delta_uri          = remote_folder,
                                                                    dimensions                  = DIMENSIONS,
                                                                    approximate_neighbors_count = 150,
                                                                    distance_measure_type       = "DOT_PRODUCT_DISTANCE",
                                                                    leaf_node_embedding_count   = 500,
                                                                    leaf_nodes_to_search_percent= 80,
                                                                    description                 = DESCRIPTION
                                                                   )

#CREATE ENDPOINT
DISPLAY_NAME = "pdf_embeddings_endpoint"
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(display_name = DISPLAY_NAME,
                                                                  description  = DESCRIPTION,
                                                                  public_endpoint_enabled = True,
                                                                  #network      = VPC_NETWORK_FULL,
                                                                 )
my_index_endpoint_str = f'projects/{my_index_endpoint.project}/locations/{my_index_endpoint.location}/indexEndpoints/{my_index_endpoint.name}'

#DEPLOY INDEX TO ENDPOINT
DEPLOYED_INDEX_ID = "pdf_st5_large_embeddings_index_deployed"
my_index_endpoint_deployed = my_index_endpoint.deploy_index(index             = tree_ah_index,
                                                            deployed_index_id = DEPLOYED_INDEX_ID)
