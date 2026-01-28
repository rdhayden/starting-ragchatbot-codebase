# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot for educational course materials. Users ask questions about courses and receive AI-powered answers with source attribution.

## Commands

**Always use `uv` to manage dependencies and run Python commands. Do not use `pip` directly.**

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh

# Or run manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

Access at http://localhost:8000 (API docs at /docs)

## Architecture

### Data Flow

1. **User query** → Frontend POSTs to `/api/query`
2. **FastAPI** (`backend/app.py`) → calls `rag_system.query()`
3. **RAGSystem** (`backend/rag_system.py`) → orchestrates the flow:
   - Retrieves conversation history from SessionManager
   - Calls AIGenerator with the search tool
4. **AIGenerator** (`backend/ai_generator.py`) → makes Claude API call
   - If Claude decides to search: executes `search_course_content` tool
   - Tool queries ChromaDB via VectorStore
   - Second Claude call with search results generates final answer
5. **Response** returns with answer + sources to frontend

### Key Components

- **`backend/rag_system.py`** - Main orchestrator connecting all components
- **`backend/ai_generator.py`** - Claude API integration with tool-calling loop
- **`backend/search_tools.py`** - Tool definitions for Claude (CourseSearchTool + ToolManager)
- **`backend/vector_store.py`** - ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunks)
- **`backend/document_processor.py`** - Parses course files into chunks with lesson metadata
- **`backend/session_manager.py`** - Conversation history per session
- **`backend/config.py`** - Central configuration (reads from `.env`)

### Course Document Format

Files in `docs/` follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
...
```

### Frontend

Vanilla HTML/CSS/JS in `frontend/`. Uses Marked.js for markdown rendering. No build step required.

## Environment

Requires `ANTHROPIC_API_KEY` in `.env` file at project root.
