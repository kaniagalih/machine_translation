# Javanese to Indonesian Machine Translation

This project implements a machine translation model that translates text from Javanese to Indonesian using Long Short-Term Memory (LSTM) networks with an attention mechanism. The model is designed to handle the unique linguistic features of both languages, providing accurate translations.

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
    mt_venv\Scripts\activate #on windows
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

