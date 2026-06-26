# Small Language Model (GPT-style) for Text Generation

A minimal GPT-style decoder-only Transformer language model trained from scratch on the TinyStories dataset. This project demonstrates the full pipeline of training, evaluating, and generating text with a small-scale language model.

---

## 📁 Project Structure

```
.
├── data_prepare.py              # Dataset download, tokenization, bin creation
├── model.py                     # GPT architecture (Transformer decoder)
├── train.py                     # Training loop with evaluation & checkpointing
├── generate.py                  # Text generation using trained model
├── requirements.txt             # Dependencies
├── data/
│   ├── train.bin               # Training dataset (binary format)
│   └── val.bin                 # Validation dataset (binary format)
├── best_model.pt               # Best checkpoint saved during training
├── loss_curve.png              # Training/validation loss plot
├── slm_generated_samples.txt    # Generated text samples
└── README.md
```

---

## 🏗️ Model Architecture

A decoder-only Transformer (GPT-style) architecture with the following specifications:

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 50,257 (GPT-2 tokenizer via tiktoken) |
| Block Size (Context Length) | 128 tokens |
| Transformer Layers | 8 |
| Attention Heads | 8 |
| Embedding Dimension | 512 |
| Dropout | 0.1 |
| **Total Parameters** | **~76.8M** |

### Key Components

- **Token & Positional Embeddings** — Learnable embeddings for tokens and positions
- **Causal Self-Attention** — Masked attention to prevent looking at future tokens
- **Feed-Forward MLP Blocks** — GELU activation functions for non-linearity
- **Layer Normalization** — Stabilizes training
- **Linear Projection** — Maps hidden states to vocabulary logits

---

## 📊 Dataset

**Source:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (HuggingFace)

**Type:** Short narrative stories designed for training small language models

### Preprocessing Pipeline

1. Download dataset from HuggingFace
2. Tokenize using GPT-2 BPE (via `tiktoken`)
3. Append end-of-text token
4. Convert to binary format (`train.bin`, `val.bin`)
5. Store using NumPy memmap for efficient memory-mapped access

---

## 🚀 Training Setup

| Configuration | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.01 |
| LR Scheduler | CosineAnnealingLR |
| Batch Size | 32 |
| Sequence Length | 128 tokens |
| Total Training Steps | 16,000 |
| Device | CUDA (GPU) |

### Training Pipeline

At each training step:

1. Sample batch from memmap dataset
2. Forward pass through GPT model
3. Compute cross-entropy loss
4. Backpropagation + optimizer step
5. Update learning rate scheduler

### Evaluation

Every 500 steps:
- Evaluate on training and validation sets
- Track loss progression
- Save best-performing model checkpoint

---

## 📈 Results

### Training Metrics

- **Initial Loss:** ~10.0
- **Final Loss:** ~1.8
- **Best Validation Loss:** ~1.83
- **Perplexity:** ~6.23
- **Best Checkpoint:** `best_model.pt`

### Model Observations

✅ Generates coherent short narratives  
✅ Maintains character consistency over short context  
✅ Learns realistic storytelling patterns  
⚠️ Occasional repetition or mild logical drift (expected for small-scale models)

---

## 📝 Text Generation

The model generates text using **autoregressive sampling**, generating one token at a time based on the previous context.

### Generation Process

1. Load trained `best_model.pt` checkpoint
2. Initialize with a prompt
3. Token-by-token generation with temperature-based sampling
4. Decode output tokens using GPT-2 tokenizer (tiktoken)
5. Apply multinomial sampling over vocabulary distribution

### Example Prompts

- `Once upon a time`
- `The little rabbit`
- `One sunny day`
- `There was a small dragon`

### Sample Generated Outputs

#### Example 1: "Once upon a time"

> Once upon a time, there was a big, friendly bear. He loved to play in the forest and collect nuts. He put them in his sack and climbed a tree to get some. But when he reached the top, he realized that he wasn't allowed to climb the tree because he had to fall.
>
> The bear was sad and didn't know what to do. He thought and thought for a moment, but he couldn't. He decided to ask his friend, the squirrel, for help.

#### Example 2: "The little rabbit"

> The little rabbit was very happy and couldn't wait to play with his new friend. Once upon a time, there was a little girl named Lily. One day, she went to the park to play with her friends. While they were playing, Lily's mom noticed that she was hurt.

#### Example 3: "One sunny day"

> One sunny day, the little duck went to the pond. The pond was huge and had lots of fish. The little duck saw a big fish and wanted to swim to it. The little duck was scared and didn't know what to do.
>
> But then, a friendly duck called out to the little duck. The duck said, "Don't be afraid, little duck. I will teach you how to swim."

#### Example 4: "There was a small dragon"

> There was a small dragon. He was very friendly and curious. He wanted to explore the world but was scared. He asked a wise dragon for help, and the dragon told him how to stay safe while exploring.

---

## 🛠️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
python data_prepare.py
```

This downloads and preprocesses the TinyStories dataset.

### 3. Train Model

```bash
python train.py
```

**Outputs:**
- Training and validation loss logs
- `best_model.pt` checkpoint
- `loss_curve.png` visualization

### 4. Generate Text

```bash
python generate.py
```

Generates new text using the trained model.

---

## 📦 Requirements

```
torch
numpy
datasets
tiktoken
matplotlib
tqdm
```

Install all at once with:
```bash
pip install -r requirements.txt
```

---

## ✨ Summary

This project demonstrates a complete pipeline for training a small-scale GPT-style language model:

```
Data Preprocessing → Transformer Training → Evaluation → Autoregressive Text Generation
```

The model successfully learns the statistical patterns of storytelling and generates coherent, narrative-driven text sequences.

---

**Created with PyTorch & Transformers** 🔥
