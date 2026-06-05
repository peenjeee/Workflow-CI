# Workflow CI Advanced

- Folder `MLProject` berisi `MLproject`, `conda.yaml`, `modelling.py`, dan dataset siap latih.
- Workflow `.github/workflows/mlflow-ci.yml` menjalankan MLflow Project.
- Artifact MLflow diunggah ke GitHub Actions artifact.
- Docker image dibuat dengan `mlflow models build-docker` dan dipush ke Docker Hub.
