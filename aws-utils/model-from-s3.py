import os
import boto3
import botocore
from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.environ.get('aws_access_key_id')
aws_secret_access_key = os.environ.get('aws_secret_access_key')

s3_client = boto3.client('s3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
)

bucket_name = os.environ.get('bucket_name')
object_name = os.environ.get('object_name')
file_path = '../models/massive-us-en.pt'

try:
    s3_client.download_file(bucket_name, object_name, file_path)
    print('Model downloaded successfully.')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == '404':
        print('The object does not exist.')
    else:
        raise



