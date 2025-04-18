import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/coldstrel/mlops_project.mlflow")

import dagshub
dagshub.init(repo_owner='coldstrel', repo_name='mlops_project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
