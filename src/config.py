import yaml

from src.s3_download import download_from_s3

with open("./config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

MODEL_PATH  = cfg['MODEL_PATH']
CONFIG_PATH = cfg['CONFIG_PATH']
CONFIG = cfg
def load_model():

    # download_from_s3(cfg['DOWNLOAD_MODEL_PATH'])
    # download_from_s3(CONFIG_PATH)
    # download_from_s3(cfg['RNN_PATH'])

    print("Model Downloaded from S3")



