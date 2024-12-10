# Javanese to Indonesian Machine Translation

This project implements a machine translation model that translates text from Javanese to Indonesian using Bidirectional Long Short-Term Memory (Bi-LSTM) networks with an attention mechanism. The model is designed to handle the unique linguistic features of both languages, providing accurate translations.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/kaniagalih/machine_translation.git
    ```

2. **Create a Virtual Environment**
    ```bash
    python3 -m venv mt_venv #on windows & linux 
    conda env create --prefix ./mt_venv --file requirements.yaml #using conda env 
    ```

3. **Activate the Virtual Environment**

    ```bash
    mt_venv/Scripts/activate #on windows
    conda activate ./mt_venv #using conda
    source .mt_venv/bin/activate #on linux 
    ```

4. **Install the Required Packages**
    ```bash
    pip install -r requirements.txt
    ```

6. **Deactive Environment**
    ```bash
    conda deactivate #conda 
    deactivate #windows & linux 
    ```

## Running the Application

### Option 1: Run Using Streamlit

Run the application using Streamlit:

    ```bash
    streamlit run app/dev/app.py
    ```

### Option 2: Run Using Docker

If you'd like to deploy the application using Docker, follow these steps:

1. **Build the Docker Image**
    ```bash
    docker build -t java-translator . | docker-compose up -d 
    ```

2. **Run the Docker Container**
    ```bash
    docker run -d -p 8501:8501 java-translator | if you do "docker-compose up -d" you don't need to run this command 
    ```

3. **Access the Application**
    After running the container, open your browser and go to the following URL:
    ```bash
    http://localhost:8501/
    ```
4. **Stop the Docker**
    ```bash
    docker stop | docker-compose down
    ```