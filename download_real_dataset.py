"""
Download and prepare the real Davidson hate-speech dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("DOWNLOADING REAL HATE-SPEECH DATASET")
print("=" * 60)

# Download the dataset
url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"

print(f"\n📥 Downloading from: {url}")
print("   Please wait...")

try:
    df = pd.read_csv(url)
    print(f"✅ Downloaded successfully!")
    print(f"   Total samples: {len(df):,}")
    
    # Show dataset structure
    print(f"\n📊 Dataset structure:")
    print(df.head())
    
    print(f"\n📋 Columns: {list(df.columns)}")
    
    # The Davidson dataset has these columns:
    # - class: 0 = hate speech, 1 = offensive language, 2 = neither
    # - tweet: the text content
    
    print(f"\n🏷️  Original label distribution:")
    print(df['class'].value_counts().sort_index())
    print(f"\n   0 = hate speech")
    print(f"   1 = offensive language")
    print(f"   2 = neither (neutral)")
    
    # Map to our label format
    label_mapping = {
        0: 'hate_speech',
        1: 'offensive',
        2: 'neutral'
    }
    
    # Create processed dataset
    processed_df = pd.DataFrame({
        'text': df['tweet'],
        'label': df['class'].map(label_mapping)
    })
    
    # Remove any NaN values
    processed_df = processed_df.dropna()
    
    print(f"\n🔄 Processed label distribution:")
    print(processed_df['label'].value_counts())
    
    # Save to data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / "davidson_hate_speech.csv"
    processed_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved to: {output_path}")
    print(f"   Total samples: {len(processed_df):,}")
    
    # Show some examples
    print(f"\n📝 Sample texts:")
    print("\nHate Speech examples:")
    hate_samples = processed_df[processed_df['label'] == 'hate_speech']['text'].head(3)
    for i, text in enumerate(hate_samples, 1):
        print(f"   {i}. {text[:80]}...")
    
    print("\nOffensive examples:")
    offensive_samples = processed_df[processed_df['label'] == 'offensive']['text'].head(3)
    for i, text in enumerate(offensive_samples, 1):
        print(f"   {i}. {text[:80]}...")
    
    print("\nNeutral examples:")
    neutral_samples = processed_df[processed_df['label'] == 'neutral']['text'].head(3)
    for i, text in enumerate(neutral_samples, 1):
        print(f"   {i}. {text[:80]}...")
    
    print("\n" + "=" * 60)
    print("✅ DATASET READY FOR TRAINING!")
    print("=" * 60)
    
    print(f"\n🚀 Next step: Train the model")
    print(f"   Run: python train.py --data {output_path}")
    print(f"\n⏱️  Expected training time:")
    print(f"   • On CPU: ~30-45 minutes")
    print(f"   • On GPU: ~8-12 minutes")
    
    print(f"\n📊 This real dataset will give you:")
    print(f"   ✅ Accurate hate-speech detection")
    print(f"   ✅ Meaningful predictions")
    print(f"   ✅ Real-world performance")
    print(f"   ✅ Benchmark-quality results")
    
except Exception as e:
    print(f"\n❌ Error downloading dataset: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Check internet connection")
    print("   2. Verify URL is accessible")
    print("   3. Run: pip install pandas")
    import traceback
    traceback.print_exc()
