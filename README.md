# TextSummarizer Using Huggingface

### Workflows 
- We are trying to create our project in a form which will be scalable in the production. (Not like in Jupyter notebook)

1. Config.yaml -> Basic Configurations and all we need to update.
2. Params.yaml -> Whenever we are doing model training we need to update all the parameters that is required for our models. (Any parameters -> Hyperparameter and all).
3. Config entity
4. Configuration Manager
5. Update the components -> Data Ingestion, Data Transformation, Model Trainer etc. components are there.
6. Create our Pipeline --> Training Pipeline, Prediction Pipeline
7. Then we will create our Frontend --> API's, Training API's, Batch Prediction API's

- Every module we define we have to follow the same process as listed above.