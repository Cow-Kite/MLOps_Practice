# Kubeflow pipeline 설치

```bash
export PIPELINE_VERSION=1.8.5

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"

kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```


Refs : https://fmind.medium.com/how-to-install-kubeflow-on-apple-silicon-3565db8773f3