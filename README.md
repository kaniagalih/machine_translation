# Javanese to Indonesian Machine Translation

This project implements a machine translation model that translates text from Javanese to Indonesian using Long Short-Term Memory (LSTM) networks with an attention mechanism. The model is designed to handle the unique linguistic features of both languages, providing accurate translations.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/kaniagalih/machine_translation.git
    ```

2. **Create a Virtual Environment**
    ```bash
    python -m venv mt_venv
    ```
    or
    ```bash
    conda env create --prefix ./mt_venv --file requirements.yaml
    ```

3. **Activate the Virtual Environment**

    ```bash
    mt_venv\Scripts\activate
    ```
    or 
     ```bash
   conda activate ./mt_venv
    ```

4. **Install the Required Packages**
    ```bash
    pip install -r requirements.txt
    ```
5. **Verify Installation**
    ```bash
    conda list
    ```
6. **Deactive Environment**
     ```bash
    conda deactivate
    ```