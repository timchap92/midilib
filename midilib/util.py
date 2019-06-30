import tqdm
from google.cloud import storage


def noop_tqdm(iterable, *args, **kwargs):
    return iterable


registered_tqdm = noop_tqdm


def register_tqdm(tqdm):
    global registered_tqdm
    registered_tqdm = tqdm


def _get_blob(path):
    assert path.startswith('gs://')  #

    path = path[5:]
    bucket_name, *blobpath = path.split('/')
    blobpath = '/'.join(blobpath)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return bucket.blob(blobpath)


def dump_string_to_gcs(s, path):
    blob = _get_blob(path)
    blob.upload_from_string(s)


def load_string_from_gcs(s, path):
    blob = _get_blob(path)
    return blob.download_as_string().decode('utf-8')
