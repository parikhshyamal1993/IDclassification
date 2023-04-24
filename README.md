# ID Classification Application
This application is training on custom dataset prepared by mining images form internet.
Source code includes training and inference code .

# Create Environment

python -m venv ClassifyEnv
source ./ClassifyEnv/bin/activate

# Training

1) place images to separate folders in dataset space
2) change label2idx dict based on directry structure
4) Run:
    ```
    python3 Trainer.py
    ```
# Inference

1) Run:
    ```
    python3 Application.py
    streamlit run ClassificationApp.py
    ```
