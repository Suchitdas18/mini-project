---
title: Hate-Speech Detection System
emoji: 🛡️
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 4.11.0
app_file: app_gradio.py
pinned: false
license: mit
---

# 🛡️ Hate-Speech Detection System

AI-powered hate speech detection with continual learning capabilities.

## 🎯 Features

- **Real-time Detection**: Instant classification of text into three categories:
  - 🟢 **Neutral**: Safe, respectful content
  - 🟡 **Offensive**: Potentially offensive language
  - 🔴 **Hate Speech**: Hateful or discriminatory content

- **Confidence Scores**: Get probability distributions for all categories
- **Modern Interface**: Clean, intuitive Gradio UI
- **Fast Inference**: Powered by RoBERTa transformer model

## 🚀 Quick Start

Simply enter text in the input box and click "Analyze" to get predictions!

### Example Texts

Try these examples to see how the model works:

**Hate Speech:**
- "You're worthless trash"
- "Get out of here, nobody wants you"

**Offensive:**
- "This is stupid"
- "What a waste of time"

**Neutral:**
- "Great work on the project!"
- "Thanks for your help"

## 🔧 Model Details

- **Base Model**: RoBERTa (roberta-base)
- **Fine-tuned on**: Hate speech dataset
- **Classes**: 3 (Neutral, Offensive, Hate Speech)
- **Max Sequence Length**: 512 tokens
- **Framework**: PyTorch + Transformers

## 📊 Use Cases

- Content moderation for social media
- Comment filtering for websites
- Community safety monitoring
- Educational tool for understanding toxic language

## ⚠️ Limitations

- This is an AI model and may not be 100% accurate
- Context and cultural nuances may affect predictions
- Always use human judgment for final moderation decisions
- Model performance may vary on different text types

## 🔗 Resources

- [GitHub Repository](https://github.com/Suchitdas18/mini-project)
- [Technical Documentation](https://github.com/Suchitdas18/mini-project/blob/main/technical_specification.md)
- [Project Summary](https://github.com/Suchitdas18/mini-project/blob/main/PROJECT_SUMMARY.md)

## 📝 Citation

If you use this model in your research, please cite:

```bibtex
@misc{hate-speech-detection-2024,
  author = {Suchit Das},
  title = {Hate-Speech Detection with Continual Learning},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/spaces/YOUR_USERNAME/hate-speech-detector}}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please check the GitHub repository for contribution guidelines.

---

Built with ❤️ using Gradio, Transformers, and PyTorch
