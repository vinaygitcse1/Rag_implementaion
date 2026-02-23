# 🤖 Simple RAG Model: Agentic Chatbot System

A full-stack Retrieval-Augmented Generation (RAG) project that combines the power of Large Language Models (LLMs) with real-time document retrieval. This system features a modular agent-based backend for processing and evaluating data, connected to a fast, modern frontend chatbot interface.

## ✨ Key Features
* **🧠 Retrieval-Augmented Generation:** Seamlessly combines LLMs with real-time document retrieval (via ChromaDB) to deliver context-rich, accurate answers.
* **🤖 Modular Agentic Backend:** Utilizes specialized custom agents for distinct tasks: scraping (Crawl4AI), document retrieval, and response evaluation.
* **💬 Interactive Chatbot UI:** A clean, responsive, and dynamic user interface built with Next.js and Tailwind CSS for real-time querying.
* **📊 Automated Evaluation:** Features an integrated evaluator agent that tracks RAG performance using custom JSON metrics to ensure high-quality responses.

## 🛠️ Technology Stack

**Frontend (Chatbot UI):**
* Framework: Next.js, React
* Language: TypeScript
* Styling: Tailwind CSS

**Backend (Agentic Pipeline):**
* Language: Python
* Vector Database: ChromaDB
* Web Scraping: Crawl4AI
* Architecture: Custom Agent Orchestration

## 📂 Project Structure

The project is divided into two main directories:

### `backend/`
* `rag.py`: Core logic for the RAG pipeline.
* `document_agent.py`: Handles document ingestion and vector retrieval.
* `scrape_agent.py`: Scrapes and processes external data sources.
* `agent_communication.py`: Manages the flow and communication between various agents.
* `evaluator.py`: Automated evaluation of RAG responses and metric tracking.
* `chroma_store/`: Local directory for the Chroma vector database.
* `requirements.txt`: Python dependencies.

### `frontend/` (chatbot/)
* `src/app/`: Main application logic, chatbot page, and dynamic routing.
* `public/images/`: UI assets and graphics.
* `screenshots/`: Demo images and project usage examples.

## ⚙️ How It Works (System Architecture)
1. **Query Submission:** The user submits a question via the Next.js chatbot interface.
2. **Semantic Retrieval:** Backend agents use ChromaDB to perform a semantic search and retrieve highly relevant documents.
3. **Generation:** The LLM synthesizes a response using both the retrieved context and its foundational knowledge.
4. **Quality Check:** The Evaluator agent scores the generated response and updates system metrics.
5. **Delivery:** The context-aware, evaluated answer is instantly displayed to the user in the UI.

## 🧩 Challenges & Solutions
* **Agent Coordination:** Designed robust communication protocols to ensure smooth data handoffs between scraping, retrieval, and evaluation agents.
* **Semantic Search Efficiency:** Successfully integrated ChromaDB for fast, vector-based document retrieval.
* **Quality Assurance:** Overcame LLM hallucination by building custom JSON metrics and automated evaluation tools to score RAG quality.
* **Full-Stack Integration:** Ensured a seamless and real-time connection between the React-based frontend UI and the Python agentic backend.


