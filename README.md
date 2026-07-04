---
title: Text Summarization
emoji: 📝
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - text-summarization
  - nlp
  - transformers
  - t5
  - fastapi
pinned: false

---

# 📝 Text Summarization

A fast, AI-powered text summarization web app built with **FastAPI** and a fine-tuned **T5** model ([Yag06/text_summarization](https://huggingface.co/Yag06/text_summarization)).

## 🚀 Features

- **Instant Summaries** — paste any text and get a concise AI-generated summary in seconds
- **T5 Transformer Model** — fine-tuned on dialogue/conversation datasets for high-quality output
- **Beautiful UI** — modern glassmorphism interface with smooth animations
- **Health Check** — real-time model status indicator

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Model | T5 (via HuggingFace Transformers) |
| Frontend | Vanilla HTML/CSS/JS |
| Container | Docker (Python 3.10-slim) |

## 📖 Usage

1. Open the app
2. Paste or type the text you want summarized
3. Click **Summarize**
4. Read your AI-generated summary!

## 📡 API

**POST** `/summarize/`

```json
{
  "dialogue": "Your text to summarize goes here..."
}
```

**Response:**

```json
{
  "summary": "AI-generated summary."
}
```

**GET** `/health` — returns model status and device info.

## 🤖 Model

- **Model ID**: [Yag06/text_summarization](https://huggingface.co/Yag06/text_summarization)
- **Architecture**: T5 (Text-to-Text Transfer Transformer)
- **Task**: Abstractive Text Summarization
