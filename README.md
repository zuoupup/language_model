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

## **1. Dataset Generation (`dataset.py`)**

The custom dataset consists of 10,000 sentences involving humans and animals performing various actions. The sentences are logically consistent, ensuring that only realistic combinations of subjects, verbs, and objects appear in the dataset.

### **Examples of Sentences Generated**

- "The boy runs in the park quickly."
- "A cat sleeps peacefully on the sofa."
- "An artist paints a landscape beautifully."
- "A dog jumps over the fence happily."

### **How It Works**

- **Subjects**: Divided into two categories: human subjects (e.g., "The boy", "A girl", "An engineer") and animal subjects (e.g., "The cat", "A dog").
- **Verbs**: Different sets of transitive (e.g., "eats", "reads", "writes") and intransitive verbs (e.g., "sleeps", "runs", "jumps") are associated with these subjects.
- **Objects and Adverbs**: Each verb can have associated objects and adverbs, ensuring logical consistency (e.g., "eats an apple", "writes a report", "sleeps peacefully").

The dataset is saved in a file named **`custom_dataset.txt`**.

### **Usage**

To generate the dataset, simply run:

```bash
python dataset.py
