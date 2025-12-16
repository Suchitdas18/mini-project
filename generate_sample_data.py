"""
Sample dataset generator for demonstration purposes
Creates a synthetic hate-speech detection dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_dataset(num_samples: int = 5000, save_path: str = "data/sample_data.csv"):
    """
    Generate a sample hate-speech detection dataset
    
    Args:
        num_samples: Number of samples to generate
        save_path: Path to save the dataset
    """
    np.random.seed(42)
    
    # Sample texts for each category
    neutral_templates = [
        "I really enjoyed the movie",
        "The weather is nice today",
        "Looking forward to the weekend",
        "This book is interesting",
        "Great work on the project",
        "Thanks for your help",
        "Have a good day",
        "That's a good point",
        "I agree with your perspective",
        "Nice to meet you",
    ]
    
    offensive_templates = [
        "This is stupid",
        "You're annoying",
        "What a waste of time",
        "This sucks",
        "That's dumb",
        "Get lost",
        "Whatever",
        "This is trash",
        "So lame",
        "Knock it off",
    ]
    
    hate_speech_templates = [
        "You're worthless trash",
        "People like you are disgusting",
        "You should be ashamed of yourself",
        "Get out of here, nobody wants you",
        "You're a waste of space",
        "Horrible excuse for a person",
        "You make me sick",
        "Pathetic loser",
        "Complete failure as a human",
        "Absolutely despicable behavior",
    ]
    
    texts = []
    labels = []
    
    # Generate samples with class balance
    samples_per_class = num_samples // 3
    
    # Neutral samples
    for _ in range(samples_per_class):
        text = np.random.choice(neutral_templates)
        texts.append(text)
        labels.append("neutral")
    
    # Offensive samples
    for _ in range(samples_per_class):
        text = np.random.choice(offensive_templates)
        texts.append(text)
        labels.append("offensive")
    
    # Hate speech samples
    for _ in range(samples_per_class):
        text = np.random.choice(hate_speech_templates)
        texts.append(text)
        labels.append("hate_speech")
    
    # Create dataframe
    df = pd.DataFrame({
        "text": texts,
        "label": labels,
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Generated {len(df)} samples and saved to {save_path}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts())
    
    return df


if __name__ == "__main__":
    generate_sample_dataset()
