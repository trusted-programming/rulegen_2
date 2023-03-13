import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

BUCKET_NAME = "ai4code"

ACCESS_KEY = 'AKIA3VN476GJNA2XN3MN'
SECRET_KEY = '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'


def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    try:
        with tqdm(unit='bytes', unit_scale=True, unit_divisor=1024, miniters=1, desc=file_name) as progress:
            s3_client.upload_file(
                file_name,
                bucket,
                object_name,
                Callback=progress.update,
                Config=boto3.s3.transfer.TransferConfig(
                    multipart_threshold=1024 * 25,
                    max_concurrency=10,
                    multipart_chunksize=1024 * 25,
                    use_threads=True
                )
            )
    except ClientError as e:
        logging.error(e)
        return False

    return True

upload_file("../checkpoints_hole_in_context_length_1024/codet5-base_epoch_91.zip", BUCKET_NAME, "codet5-mising-rule-prediction.zip")