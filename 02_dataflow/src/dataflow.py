# Beam Imports
import apache_beam as beam
from apache_beam.utils import shared

# Python Imports
import argparse
from datetime import datetime

def get_args():
    '''Function to parse arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_bq_table', help='Input BQ table', default='leo-gcp-sanbox.amazon_products.books')
    parser.add_argument('--mode', choices=['local', 'cloud'], help='Run the job Locally or in Cloud Dataflow.', default='local')
    parser.add_argument('--model_path', help='Model location', default='gs://leo-models/fasttext_pretrained')
    parser.add_argument('--output_bq_table', help='Output BQ table', default='leo-gcp-sanbox.amazon_products.books_title_embeddings')

    return parser.parse_args() 


class FTEmbGen(beam.DoFn):
    def __init__(self, shared_handle, model_path):
        self.shared_handle = shared_handle
        self.model_path = model_path
        
    def setup(self):
        '''Load the model into memory once per machine using shared_handle'''
        
        def initialize_model():
            '''Generic python function to load the model into memory'''
            import fasttext
            import subprocess
            import os

            # Download the model directory from GCS bucket to the current directory
            subprocess.run(['gsutil', 'cp', '-r', self.model_path, '.'])
            downloaded_model_path = self.model_path.split('/')[-1]

            return fasttext.load_model(
                os.path.join(downloaded_model_path, 'cc.en.300.bin')
            )
        
        # Use shared_handle to enzure model is loaded only once
        self.ft_model = self.shared_handle.acquire(initialize_model)
            
    def process(self, rowdict):
        # run the model to generate sentence embedding vector on the "title" key of rowdict
        sentence_vector = self.ft_model.get_sentence_vector(rowdict['title']).tolist()
        
        # update rowdict with the embedding results
        rowdict.update({f'FT_emb_{i}': emb for i,emb in enumerate(sentence_vector)})
        
        # yield the processed rowdict
        yield rowdict
    
            
def main(args):
    '''The main processing function
    '''
    
    # Read the SQL query
    with open('dataquery.sql') as f:
        query = f.read()

    # Handle "input_bq_table" argument by sending it to the query
    query = query.format(**vars(args))

    # Handle "output_bq_table" argument
    output_project, output_dataset, output_table = args.output_bq_table.split('.')

    # Specify Output table schema: 1 title column and 300 columns for fasttext embedding
    output_schema = 'title:STRING'
    for i in range(300):
        output_schema += f',FT_emb_{i}:FLOAT'


    # Initialize Pipeline
    ## Handle "mode" argument
    if args.mode=='local':
        query = f'{query} LIMIT 100'
        runner = 'DirectRunner'

    elif args.mode=='cloud':
        query = f'{query} LIMIT 1000'
        runner = 'DataflowRunner'

    else:
        raise ValueError(f'"mode" argument should be either "local" or "cloud". "{args.mode}" is not allowed')

    ## Beam pipeline options
    opts = {
        'project': 'leo-gcp-sanbox',
        'job_name': 'amazon-books-ft-emb-gen-' + datetime.now().strftime('%y%m%d-%H%M%S'),
        'temp_location': 'gs://leo-dataflow/tmp',
        'setup_file': './setup.py',
        'region': 'us-central1',
        'max_num_workers': 20
      }

    options = beam.pipeline.PipelineOptions(flags=[], **opts)

    ## Instantiate the Pipeline
    with beam.Pipeline(runner, options) as p:

        ### Shared handle instantiation
        shared_handle = shared.Shared()

        ### Pipeline Code
        pipeline = ( 
            p
            | 'Query Data from BQ' >> beam.io.ReadFromBigQuery(query=query, use_standard_sql=True)
            | 'FT Embeddings Gen' >> beam.ParDo(FTEmbGen(shared_handle, args.model_path))
            | 'Write FT Emb to BQ' >> beam.io.WriteToBigQuery(
                project=output_project, 
                dataset=output_dataset,
                table=output_table,
                schema=output_schema,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
            )
        )

    print('Done!')
    
        
if __name__ == '__main__':
    args = get_args()
    main(args)    