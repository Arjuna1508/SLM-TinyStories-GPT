import torch
import tiktoken

from model import GPT, GPTConfig


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("best_model.pt", map_location=device)

config = checkpoint["config"]

model = GPT(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

enc = tiktoken.get_encoding("gpt2")


def generate_text(prompts, temperature=0.7, max_new_tokens=200):

    if isinstance(prompts, str):
        prompts = [prompts]

    for prompt in prompts:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)

        tokens = enc.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_new_tokens):

                logits, _ = model(x)

                logits = logits[:, -1, :]
                logits = logits / temperature

                probs = torch.softmax(logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)

                x = torch.cat((x, next_token), dim=1)

        output = enc.decode(x[0].tolist())

        print(output)
        print("\n")


prompts = [
    "Once upon a time",
    "The little rabbit",
    "One sunny day",
    "There was a small dragon"
]


generate_text(prompts, temperature=0.7)
generate_text(["Once upon a time"], temperature=1.0)