# Knowledge Transfer Agent

## Overview

The Knowledge Transfer Agent is a web-based application designed to facilitate smooth onboarding for new employees by capturing and sharing role-specific knowledge from outgoing employees. It uses AI-powered summarization and question-answering to create comprehensive role summaries based on initial descriptions and supporting documents. New employees can then access these summaries and ask follow-up questions, with answers grounded in the provided knowledge base using Retrieval-Augmented Generation (RAG).

Key functionalities:
- **Outgoing Employee Interface**: Input job title, role summary, and upload documents (TXT, DOCX, PDF). Generate a summary without questions or start a clarification session where AI generates targeted questions to elicit details.
- **Incoming Employee Interface**: Select a role to view its summary and engage in a Q&A chat powered by embeddings and LLM for accurate, source-cited responses.
- Knowledge is stored in Supabase, with embeddings for semantic search.

This tool reduces knowledge loss during employee transitions, especially in HR and team onboarding processes.

## Features

- **Document Processing**: Extract text from TXT, DOCX, and PDF files.
- **AI-Driven Clarification**: Automatically generate 5-8 targeted questions based on role ambiguities (e.g., processes, responsibilities).
- **Role Summarization**: Create concise summaries incorporating initial input, documents, and Q&A clarifications using Google Gemini.
- **RAG-Powered Q&A**: Semantic search over chunked documents and summaries for precise answers, with source citations (clickable links to documents).
- **Persistence**: Store roles, embeddings, Q&A pairs, and interactions in Supabase for versioning and reuse.
- **User-Friendly UI**: Built with Gradio for an interactive chat-like experience, with tabs for outgoing/incoming users.
- **Caching & Efficiency**: Reuse roles and embeddings if inputs unchanged; batch embedding generation.
- **Fallback Mechanisms**: Keyword search alongside semantic similarity for better recall.

## Tech Stack

- **Backend**: Python with FastAPI (for API endpoints) and Gradio (for UI).
- **AI/ML**:
  - LLM: Google Gemini (via `langchain-google-genai`).
  - Embeddings: Sentence Transformers (`all-MiniLM-L6-v2` model).
  - Text Chunking: LangChain's `RecursiveCharacterTextSplitter`.
- **Database**: Supabase (PostgreSQL with vector support for embeddings).
- **Document Handling**: `python-docx` for DOCX, `PyPDF2` for PDF.
- **Other**: `numpy` for vector operations, `hashlib` for caching, `dotenv` for env vars.

## Prerequisites

- Python 3.8+.
- A Supabase account and project (free tier sufficient).
- Google API key for Gemini.
- Optional: Local documents for testing.

## Setup Instructions

1. **Clone the Repository**:
   ```
   git clone https://github.com/MuhammadMusabYaqoob/Knowledge-Transfer-Agent.git
   cd Knowledge-Transfer-Agent
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file in the root directory with your credentials:
   ```
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   GOOGLE_API_KEY=your_google_api_key
   ```
   - Get Supabase details from your project dashboard (Settings > API).
   - Get Google API key from Google AI Studio (enable Generative Language API).

4. **Database Schema**:
   In your Supabase SQL Editor, run the following to create required tables (enable the `pgvector` extension first via Dashboard > Database > Extensions):

   ```sql
   -- Enable vector extension
   CREATE EXTENSION IF NOT EXISTS vector;

   -- Roles table
   CREATE TABLE IF NOT EXISTS roles (
       id BIGSERIAL PRIMARY KEY,
       job_title TEXT NOT NULL,
       initial_summary TEXT NOT NULL,
       generated_summary TEXT,
       clarification_qa JSONB DEFAULT '[]'::JSONB,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Embeddings table (with vector column)
   CREATE TABLE IF NOT EXISTS embeddings (
       id BIGSERIAL PRIMARY KEY,
       role_id BIGINT REFERENCES roles(id) ON DELETE CASCADE,
       chunk_text TEXT NOT NULL,
       embedding VECTOR(384),  -- Dimension for all-MiniLM-L6-v2
       metadata JSONB,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Interactions table (optional, for logging Q&A)
   CREATE TABLE IF NOT EXISTS interactions (
       id BIGSERIAL PRIMARY KEY,
       role_id BIGINT REFERENCES roles(id),
       question TEXT NOT NULL,
       answer TEXT NOT NULL,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Index for faster similarity search
   CREATE INDEX IF NOT EXISTS embeddings_role_idx ON embeddings(role_id);
   CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops);
   ```

5. **Run the Application**:
   ```
   python app.py
   ```
   - The app will start at `http://127.0.0.1:7860`.
   - Access the Gradio interface in your browser.

## Usage

### Outgoing Employee (Knowledge Provider)
1. Go to the "Outgoing Employee" tab.
2. Enter the **Job Title** (e.g., "Software Engineer").
3. Provide an **Initial Summary** of responsibilities and processes.
4. Upload supporting documents (optional).
5. Click **Generate Summary Without Questions** for a quick AI-generated summary.
   - Or, click **Start Clarification Session** to begin Q&A: Answer AI-generated questions one by one.
6. Once complete, the full summary (with clarifications) is saved to Supabase.

### Incoming Employee (New Hire)
1. Go to the "Incoming Employee" tab.
2. Select your **Job Title** from the dropdown (populated from saved roles).
3. View the **Role Summary**.
4. Click **Open Q&A Chat** to ask questions.
   - Type a question (e.g., "What are the daily check steps?").
   - Get responses with citations to sources (e.g., [source:HR_policy.pdf]).

### API Endpoints
- The app mounts Gradio at `/`, but you can access document files via `/docs/{filename}` (TXT only, for citations).

## Documenting Sources
- Uploaded documents are chunked and embedded.
- Q&A responses cite sources as clickable links (e.g., to view full docs).
- Only TXT files are served via API; others are processed internally.

## Limitations & Improvements
- **Embeddings**: Fixed model; consider upgrading to larger models for better accuracy.
- **Question Generation**: Limited to 5-8 questions; based solely on input to avoid hallucinations.
- **Search**: Combines cosine similarity (top 25 chunks) + keyword fallback; tunable via code.
- **Scalability**: For large docs, chunk size (300) and overlap (100) can be adjusted in `utils/chunking.py`.
- **Security**: Env vars keep keys safe; Supabase RLS (Row Level Security) recommended for production.
- **No External Knowledge**: LLM strictly uses provided context to prevent biases.

## Troubleshooting
- **API Key Errors**: Ensure `.env` is loaded and keys are valid.
- **Supabase Connection**: Check URL/key; verify tables exist.
- **Document Extraction**: PDF/DOCX may have formatting issues; test with simple files.
- **Embeddings Dimension**: Must match model (384 for MiniLM); update DB if changing model.
- **Port Conflicts**: Change `port=7860` in `app.py` if needed.

Suggestions welcome for enhancements like multi-language support, advanced RAG, or integration with HR tools.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [Gradio](https://gradio.app/), [LangChain](https://langchain.com/), [Supabase](https://supabase.com/), and [Google Gemini](https://ai.google.dev/).
- Inspired by knowledge transfer challenges in remote teams.