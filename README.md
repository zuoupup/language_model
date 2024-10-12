#Language Model with Transformer
The model is able to run successfully and generate 10 different coherent and unambiguous sentences. 

This project implements a language model using a Transformer architecture. The model is capable of generating coherent sentences based on a custom dataset, and it has been trained to recognize various patterns and structures in English sentences. The project is divided into three main files: dataset.py, model.py, and training.py.

Project Structure
dataset.py: Responsible for generating a custom dataset with 10,000 logically consistent and grammatically correct sentences.
model.py: Defines the Transformer-based language model architecture, including embedding, positional encoding, and transformer encoder layers.
training.py: Trains the model using the dataset generated from dataset.py, and evaluates its performance. It also includes functionality for generating sentences using the trained model.
Requirements
Before running the code, make sure to install the necessary Python libraries:

pip install torch scikit-learn matplotlib

1. Dataset Generation (dataset.py)
The custom dataset consists of 10,000 sentences involving humans and animals performing various actions. The sentences are logically consistent, ensuring that only realistic combinations of subjects, verbs, and objects appear in the dataset.

Examples of Sentences Generated
"The boy runs in the park quickly."
"A cat sleeps peacefully on the sofa."
"An artist paints a landscape beautifully."
"A dog jumps over the fence happily."
How It Works
Subjects: Divided into two categories: human subjects (e.g., "The boy", "A girl", "An engineer") and animal subjects (e.g., "The cat", "A dog").
Verbs: Different sets of transitive (e.g., "eats", "reads", "writes") and intransitive verbs (e.g., "sleeps", "runs", "jumps") are associated with these subjects.
Objects and Adverbs: Each verb can have associated objects and adverbs, ensuring logical consistency (e.g., "eats an apple", "writes a report", "sleeps peacefully").
The dataset is saved in a file named custom_dataset.txt.

Usage
To generate the dataset, simply run:

python dataset.py

2. Model Architecture (model.py)
The model is implemented using a Transformer architecture, leveraging PyTorch's nn.TransformerEncoder module. It includes:

Embedding Layer: Encodes the input tokens into dense vectors.
Positional Encoding: Adds positional information to the embeddings, as the transformer does not inherently know the order of tokens.
Transformer Encoder Layers: Stacks multiple layers of self-attention and feedforward networks to capture relationships between words.
Decoder: Maps the output of the transformer layers to the vocabulary size for prediction.
Positional Encoding
The positional encoding is implemented using sine and cosine functions, allowing the model to incorporate sequence information into the embeddings.

3. Training the Model (training.py)
This script trains the language model using the custom dataset generated in dataset.py. It also includes validation and generates loss plots to visualize training progress.

Training Details
DataLoader: The dataset is split into training and validation sets (90% training, 10% validation).
Training Loop: The model is trained for 200 epochs with a learning rate scheduler that reduces the learning rate if the validation loss plateaus.
Loss Function: Cross-entropy loss is used, and padding tokens are ignored during computation.
Optimization: Adam optimizer is employed, with gradient clipping to prevent exploding gradients.
Usage
To train the model:

python training.py

This will start the training process and display the loss for each epoch. The training and validation loss over epochs will be plotted once the training is complete.

Generating Sentences
The generate_text function in training.py can be used to generate sentences based on the trained model. It randomly selects a starting word (subject) and predicts the next words based on the learned patterns.

Examples of Generated Sentences
After training, the model is capable of generating sentences like:

"The cat jumps over the fence."
"My father drives a car slowly."
"An engineer designs a bridge skillfully."
These sentences are designed to be grammatically correct and logical, following the structures observed during training.

Files Overview
custom_dataset.txt: The generated dataset file containing 10,000 sentences.
dataset.py: Script for generating the dataset.
model.py: Script containing the Transformer-based model implementation.
training.py: Script for training the model, evaluating its performance, and generating sentences.
Notes
Ensure that CUDA is available if training on a GPU, as this will significantly speed up the process.
The model parameters (e.g., embedding size, number of layers) can be adjusted in training.py to experiment with different configurations.
Future Improvements
Incorporate more diverse sentence structures to expand the model's capabilities.
Implement a beam search for more coherent text generation during inference.
Fine-tune the model on larger, pre-existing text corpora for improved performance.

License
This project is licensed under the MIT License - see the LICENSE file for details.
