import pandas as pd
from sklearn.datasets import load_iris
from minio import Minio
from minio.versioningconfig import VersioningConfig, ENABLED

#
# dump data
#
iris = load_iris(as_frame=True)
X, y = iris["data"], iris["target"]
data = pd.concat([X, y], axis="columns")
data.sample(100).to_csv("iris.csv", index=None)
#
# minio client
#
url = "0.0.0.0:9000"
access_key = "minio"
secret_key = "miniostorage"
client = Minio(url, access_key=access_key, secret_key=secret_key, secure=False)
#
# upload data to minio
#
bucket_name = "raw-data"
object_name = "iris"  # 버킷이 생성되고 최초로 올리는 파일 이름
if not client.bucket_exists(bucket_name):  # 버킷이 있으면 True, 없으면 False
    client.make_bucket(bucket_name)
    client.set_bucket_versioning(bucket_name, VersioningConfig(ENABLED)) # 생성한 버킷이 버저닝이 가능하게 설정

client.fput_object(bucket_name, object_name, "iris.csv")
