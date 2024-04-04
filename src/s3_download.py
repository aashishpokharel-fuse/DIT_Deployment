import os
import boto3
from io import BytesIO
from dotenv import load_dotenv

from botocore.exceptions import NoCredentialsError

load_dotenv()
BUCKET = os.getenv('BUCKET_NAME')


def download_from_s3(model_name):

    s3 = boto3.client('s3')
    try:
        s3.download_file(BUCKET, model_name, model_name)
    except NoCredentialsError:
        print("No AWS credentials found")


def load_model_from_s3(model_name):

    Model = None
    s3 = boto3.client(
        's3',
    )
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=model_name)
        Model = BytesIO(obj['Body'].read())
    except NoCredentialsError:
        print("No AWS credentials found")

    return Model
