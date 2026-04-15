import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import tqdm
from dreem_learning_open.settings import DODH_SETTINGS, DODO_SETTINGS

# Public bucket: use unsigned requests so no AWS credentials are required.
client = boto3.client(
    "s3",
    region_name="eu-west-1",
    config=Config(signature_version=UNSIGNED),
)

bucket = "dreem-dodo-dodh"


def download_prefix(prefix, dest_dir, label):
    """Download all objects under prefix into dest_dir (flat filenames)."""
    print("\n Downloading H5 files from S3 for %s" % label)
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = resp.get("Contents") or []
    for bucket_object in tqdm.tqdm(objects):
        key = bucket_object["Key"]
        filename = key.split("/")[-1]
        if not filename:
            continue
        out_path = os.path.join(dest_dir, filename)
        client.download_file(Bucket=bucket, Key=key, Filename=out_path)


# Set to False if you only need DOD-H (saves time and bandwidth).
DOWNLOAD_DODO = True
DOWNLOAD_DODH = True

if DOWNLOAD_DODO:
    download_prefix("dod-o/", DODO_SETTINGS["h5_directory"], "DOD-O")

if DOWNLOAD_DODH:
    download_prefix("dod-h/", DODH_SETTINGS["h5_directory"], "DOD-H")
