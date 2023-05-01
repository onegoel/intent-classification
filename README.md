# Set up procedure

Clone the repository

```
git clone https://github.com/onegoel/intent-classification
```


## 1. Create a new virtual environment

```
python -m venv <name of environment>
```

## 2. Activate the virtual environment

```
source <name of environment>/bin/activate (MacOS)
```

or

```
<name of environment>\Scripts\activate.bat
```

## 3. Install the requirements

```
pip install -r dependencies.txt
```

## 4. Download model from s3 bucket

* create a .env file and paste credentials (refer to .env.example)
* run the following command

```
python model-from-s3.py
```

## 5. You can now use the Jupyter Notebook
