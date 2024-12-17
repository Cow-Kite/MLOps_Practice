import os
import mlflow
import mlflow.artifacts

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

if __name__=="__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--model-name", type=str, default="my_model")
    args = parser.parse_args()

    mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path=args.model_name, dst_path="./downloads")