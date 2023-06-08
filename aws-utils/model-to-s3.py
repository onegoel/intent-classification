import boto3
import os
from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.environ.get('aws_access_key_id')
aws_secret_access_key = os.environ.get('aws_secret_access_key')
region_name = os.environ.get('region_name')

s3_client = boto3.client('s3', 
        aws_access_key_id=aws_access_key_id, 
        aws_secret_access_key=aws_secret_access_key, 
        region_name='us-east-1')

s3_client.create_bucket(Bucket='distilbert-model-ic')

bucket_name = 'distilbert-model-ic'
object_name = 'massive-us-en.pt'
file_path = '../models/massive-us-en.pt'

s3_client.upload_file(file_path, bucket_name, object_name)
