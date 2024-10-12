# **Language Model with Transformer**

The model is able to run successfully and generate 10 different coherent and unambiguous sentences.

This project implements a language model using a Transformer architecture. The model is capable of generating coherent sentences based on a custom dataset, and it has been trained to recognize various patterns and structures in English sentences. The project is divided into three main files: `dataset.py`, `model.py`, and `training.py`.

## **Project Structure**

- **`dataset.py`**: Responsible for generating a custom dataset with 10,000 logically consistent and grammatically correct sentences.
- **`model.py`**: Defines the Transformer-based language model architecture, including embedding, positional encoding, and transformer encoder layers.
- **`training.py`**: Trains the model using the dataset generated from `dataset.py`, and evaluates its performance. It also includes functionality for generating sentences using the trained model.

## **Requirements**

Before running the code, make sure to install the necessary Python libraries:

```bash
pip install torch scikit-learn matplotlib
