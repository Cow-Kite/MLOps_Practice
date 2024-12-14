from minio import Minio

#
# minio client
#
url = "0.0.0.0:9000"
access_key = "minio"
secret_key = "miniostorage"
client = Minio(url, access_key=access_key, secret_key=secret_key, secure=False)

#
# data download
#
bucket_name = "raw-data"
object_name = "iris" 

object_stat = client.stat_object(bucket_name, object_name)
print(object_stat.version_id)
client.fget_object(bucket_name, object_name, file_path="download_data.csv")
