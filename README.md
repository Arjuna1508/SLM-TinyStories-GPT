Small Language Model (GPT-style) for Text Generation

Overview


This project implements a GPT-style decoder-only Transformer language model trained from scratch on the TinyStories dataset.
The model learns next-token prediction and generates short, coherent story-like text sequences.


## Project Structure
.
├── data_prepare.py Dataset download, tokenization, bin creation
├── model.py GPT architecture (Transformer decoder)
├── train.py Training loop with evaluation & checkpointing
├── generate.py Text generation using trained model
├── requirements.txt Dependencies
├── data/
│ ├── train.bin
│ └── val.bin
├── best_model.pt Best checkpoint saved during training
├── loss_curve.png Training/validation loss plot
├── slm_generated_samples.txt
└── README.md


Model Architecture

The model is a decoder-only Transformer (GPT-style).

Vocabulary size: 50,257 (GPT-2 tokenizer via tiktoken)
Block size: 128 tokens
Layers: 8
Attention heads: 8
Embedding dimension: 512
Dropout: 0.1


Key Components
 
Token + positional embeddings
Causal self-attention (masked attention)
Feed-forward MLP blocks (GELU activation)
Layer normalization
Linear projection to vocabulary logits

Total parameters: ~76.8M


Dataset

Dataset: TinyStories (HuggingFace)
Source: roneneldan/TinyStories
Type: short narrative stories
Preprocessing
Tokenized using GPT-2 BPE (tiktoken)
End-of-text token appended
Converted into binary format (train.bin, val.bin)
Stored using NumPy memmap for efficient loading


Training Setup

Optimizer: AdamW
Learning rate: 3e-4
Weight decay: 0.01
LR scheduler: CosineAnnealingLR
Batch size: 32
Sequence length: 128
Total steps: 16,000
Device: CUDA (GPU)


Training Behavior

Loss decreases: ~10 → ~1.8
Best validation loss: ~1.83
Perplexity: ~6.23
Best checkpoint saved as best_model.pt


Training Pipeline

At each training step:

Sample batch from memmap dataset
Forward pass through GPT model
Compute cross-entropy loss
Backpropagation + optimizer step
Update learning rate scheduler


Evaluation

Every 500 steps:

Evaluate on train and validation sets
Track loss progression
Save best-performing model checkpoint


Results

The model successfully learns statistical patterns of storytelling data.

Observations

Generates coherent short narratives
Maintains character consistency over short context
Occasional repetition or mild logical drift (expected for small-scale LM)


Text Generation

Generation uses autoregressive sampling:

Loads trained best_model.pt
Token-by-token generation
Temperature-based sampling supported
Decoding
GPT-2 tokenizer (tiktoken)
Multinomial sampling over vocabulary


Example Prompts
Once upon a time
The little rabbit
One sunny day
There was a small dragon

Sample Generated Outputs

Example 1

Prompt: Once upon a time

Once upon a time, there was a big, friendly bear. He loved to play in the forest and collect nuts. He put them in his sack and climbed a tree to get some. But when he reached the top, he realized that he wasn't allowed to climb the tree because he had to fall.

The bear was sad and didn't know what to do. He thought and thought for a moment, but he couldn't. He decided to ask his friend, the squirrel, for help.

Example 2

Prompt: The little rabbit

The little rabbit was very happy and couldn't wait to play with his new friend. Once upon a time, there was a little girl named Lily. One day, she went to the park to play with her friends. While they were playing, Lily's mom noticed that she was hurt.

Example 3

Prompt: One sunny day

One sunny day, the little duck went to the pond. The pond was huge and had lots of fish. The little duck saw a big fish and wanted to swim to it. The little duck was scared and didn't know what to do.

But then, a friendly duck called out to the little duck. The duck said, "Don't be afraid, little duck. I will teach you how to swim."

Example 4

Prompt: There was a small dragon

There was a small dragon. He was very friendly and curious. He wanted to explore the world but was scared. He asked a wise dragon for help, and the dragon told him how to stay safe while exploring.


How to Run

1. Install dependencies
pip install -r requirements.txt

2. Prepare dataset
python data_prepare.py

3. Train model
python train.py

Outputs:

model training logs
best_model.pt checkpoint
loss_curve.png plot

4. Generate text
python generate.py


Requirements
torch
numpy
datasets
tiktoken
matplotlib
tqdm

Conclusion

This project demonstrates a full pipeline for training a GPT-style language model:

data preprocessing → transformer training → evaluation → autoregressive text generation
