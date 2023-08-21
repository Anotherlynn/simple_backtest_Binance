# SignalA_up

[![Build Status](https://app.travis-ci.com/Anotherlynn/Lynns-work-file-.svg?branch=main)]()
![](https://img.shields.io/badge/powered%20by-@XinyiLi-green.svg)
![](https://img.shields.io/badge/language-Python-green.svg)
![](https://img.shields.io/badge/version-Python3.10-green.svg)
--------------------------------------------------------------------------------

The SignalA_up provides models and functions to analyse temporary announcements from companies on stock market. 

Different LLMs, embeddings methods, testing models and convenient tools, such as access to the OpenAI API, are included, written in the Python language.

You can find usage examples [here](examples.py).

[//]: # (and  [OpenAI Cookbook]&#40;https://github.com/openai/openai-cookbook/&#41;.)

## Installation

[//]: # ()
[//]: # (You don't need this source code unless you want to modify the package. If you just)

[//]: # (want to use the package, just run:)

[//]: # ()
[//]: # (```sh)

[//]: # (pip install --upgrade openai)

[//]: # (```)
 
[//]: # ()
[//]: # (Install from source with:)

[//]: # ()
[//]: # (```sh)

[//]: # (python setup.py install)

[//]: # (```)

### Creating a Environment (Optional)
> Note: You are strongly suggested to build a virtual environment in `python3.10` or `python3.9`.

To start a virtual environment, you can use [conda](https://github.com/conda/conda)
```bash
conda create -n your_env_name python=3.10
```
To activate or deactivate the enviroment, you can use:

On Linux
```bash
source activate your_env_nam
# To deactivate:
source deactivate
```
On Windows
```bash
activate your_env_name
# to deactivate
deactivate env_name # or activate root
```

### Building the Documentation
To use the tools, you need to install the packages in required version.
```bash
cd proj/
conda install -n your_env_nam requirements.txt # or pip install -r requirements.txt
```

## Getting Started

--------------------------------------------------------------------------------
> If you already have your dataset, the input file should be a dataframe with columns matching SQL db: 
<br />**[[InnerCode], [SecuCode], [BulletinDate], [InsertDate(the time for the data to be added into the database)], [InfoTitle], [Detail]]**
 <br />If not, please refer to [get_word_cloud.py](get_word_cloud.py) to crawl the data.

- Data formatting & preprocessing
  - [How to format the input dataframe and align the data scope](./proj/mergedata.py)
- Topic models
  - [How to  build a topic model on your dataset](./proj/main.py)
  - [How to customize a topic model using different embedding methods](./proj/Clustering.py)
- Embeddings
  - [How to get Glove embedding matrix](./GloVe-master)
  - [How to get embeddings of long inputs from LLMs using OpenAI api](./embedding_long_inputs.py)
- Classifier of tags
  - [Example: OpenAI API usages to get Tags (tested Prompts included)](./proj/getTAG.py)
  - [Best Practice of building a Multilabel-multioutput classifier](./proj/build_model.py)
  
- Fine-tunning of Roberta_chinese
  - [How to fine-tune a pre-trained classifier(fixing the parameters)](./proj/Transformer.py)
    - [3 Customized loss functions for unsupervised matching](./proj/UnsupervisedLoss.py)























