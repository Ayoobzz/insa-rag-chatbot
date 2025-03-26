# 🎒 INSA Chatbot

## 🌍 Overview

INSA Chatbot is a **Streamlit-based AI assistant** designed to help students with their queries about **INSA Rennes**. It leverages **LangGraph for agent orchestration**, along with **NLP processing and web scraping**, to provide accurate and up-to-date information. Currently, the chatbot integrates **Jina & GROQ APIs**, and work is in progress to expand support for additional agents.

## ✨ Features

✅ **Interactive chatbot UI** with real-time responses\
✅ **AI-powered answers** using Jina & GROQ APIs\
✅ **Vector storage** for efficient data retrieval\
✅ **LangGraph-powered agent orchestration** for modular AI expansion

## 🚀 Installation

1️⃣ Clone the repository:

```sh
git clone https://github.com/Ayoobzz/insa-rag-chatbot
cd insa-rag-chatbot
```

2️⃣ Install dependencies:

```sh
pip install -r requirements.txt
```

3️⃣ Set up your **API keys** in a `.env` file:

```sh
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
```

## 🏃 Usage

1️⃣ Run the chatbot:

```sh
streamlit run app.py
```

2️⃣ Enter your **API keys** in the sidebar (if not set in `.env`).
3️⃣ Start **chatting** and get your questions answered!

## 📂 Project Structure

```
ayoobzz-insa-rag-chatbot/
├── app.py               # Main application file
├── requirements.txt      # Dependencies
├── assets/
│   └── timetables/
│       └── timetables.json  # Stored timetable data
├── crawler/             # Web scraping components
│   ├── scraper.py       # Scrapes website data
│   ├── spider.py        # Defines crawling behavior
│   ├── utils.py         # Helper functions for crawling
├── crew/                # AI agent management
│   ├── agents.py        # Defines chatbot agents
│   ├── tasks.py         # Task execution logic
│   ├── tools.py         # Helper tools for agents
├── nlp/                 # Text processing utilities
│   ├── chunking.py      # Text chunking logic
│   ├── text_cleaner.py  # Text preprocessing functions
└── vectorstore/         # Vector storage for retrieval
    ├── utils.py         # Vector storage management
```

##

## 📜 License

This project is licensed under the **MIT License**.

## 🤝 Contributions

We welcome contributions! Feel free to **submit issues** or **pull requests** to improve the chatbot. 🚀

