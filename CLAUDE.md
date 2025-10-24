# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot** for educational course materials. It combines semantic search (ChromaDB + Sentence-Transformers) with Claude AI's tool-calling capabilities to provide intelligent, context-aware responses about course content.

**Key Architecture Pattern**: Tool-based RAG where Claude decides when to search (not hardcoded). The system uses Anthropic's tool_use API feature - Claude receives a `search_course_content` tool definition and autonomously decides whether to invoke it based on the query.

## Setup & Running

### Prerequisites
- Python 3.13+
- UV package manager
- Anthropic API key in `.env` file

### Installation
```bash
uv sync                    # Install all dependencies
cp .env.example .env       # Then add your ANTHROPIC_API_KEY
```

### Running the Application
```bash
./run.sh                   # Quick start
# OR
cd backend && uv run uvicorn app:app --reload --port 8000
```

Access at:
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Architecture

### Request Flow
```
User Query → Frontend → FastAPI → RAGSystem → AIGenerator
    → Claude API (Call #1: decides to use tool)
    → ToolManager → CourseSearchTool → VectorStore → ChromaDB
    → Claude API (Call #2: synthesizes answer from search results)
    → SessionManager (stores history) → Response to User
```

### Core Components

**RAGSystem** (`backend/rag_system.py`)
- Main orchestrator coordinating all components
- Entry point: `query(query, session_id)` - processes user queries
- Manages tool registration and source tracking
- Handles document ingestion via `add_course_document()` and `add_course_folder()`

**AIGenerator** (`backend/ai_generator.py`)
- Manages Claude API interactions with tool-calling support
- `generate_response()` - makes initial API call with tools
- `_handle_tool_execution()` - executes tools and makes follow-up API call
- **Two-phase interaction**: First call may trigger tool_use, second call synthesizes final answer

**ToolManager & CourseSearchTool** (`backend/search_tools.py`)
- Tool registry pattern: `ToolManager` registers and dispatches tools
- `CourseSearchTool.execute()` - performs semantic search with optional filters
- Tracks sources for UI display via `last_sources` attribute
- Tools follow abstract `Tool` interface for extensibility

**VectorStore** (`backend/vector_store.py`)
- Wraps ChromaDB with two collections: `course_catalog` (metadata) and `course_content` (chunks)
- Uses Sentence-Transformers for embeddings ("all-MiniLM-L6-v2")
- `search()` - unified search interface with course/lesson filtering
- Implements fuzzy course name resolution via semantic similarity

**DocumentProcessor** (`backend/document_processor.py`)
- Parses structured course documents (expects format: Course Title, Link, Instructor, then lessons)
- `chunk_text()` - sentence-based chunking with overlap (800 chars, 100 overlap)
- Smart sentence splitting that handles abbreviations
- Enriches chunks with context: "Course X Lesson Y content: ..."

**SessionManager** (`backend/session_manager.py`)
- Maintains conversation history per session (configurable MAX_HISTORY, default 2)
- History format: "User: {query}\nAssistant: {response}\n\n"
- Auto-generates UUIDs for new sessions

### Configuration

All settings in `backend/config.py`:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 800 chars
- `CHUNK_OVERLAP`: 100 chars
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges

### Data Models

Defined in `backend/models.py`:
- `Course` - course metadata (title is unique identifier)
- `Lesson` - lesson metadata within a course
- `CourseChunk` - text chunk with course/lesson context for vector storage

### Document Format

Course documents in `/docs` must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [title]
...
```

## Key Implementation Details

### Tool-Based Search Pattern
- Claude receives tool definition in system prompt
- `stop_reason: "tool_use"` indicates Claude wants to search
- Tool results are passed back to Claude in a second API call
- This allows Claude to decide whether search is needed (e.g., general questions don't require search)

### Dual Collection Strategy
- **course_catalog**: Stores course metadata for fuzzy name matching
- **course_content**: Stores chunked content for semantic search
- Enables efficient course filtering before content search

### Session State Management
- Sessions maintained server-side (not client-side)
- Conversation history sent to Claude for context-aware responses
- History limited to prevent token bloat

### Source Tracking Flow
Sources flow through: `VectorStore` → `CourseSearchTool.last_sources` → `ToolManager.get_last_sources()` → `RAGSystem` → API response → Frontend UI

### Chunking Strategy
- Sentence-based (preserves semantic units)
- Overlapping (maintains context continuity)
- Enriched with metadata (enables filtering)
- Regex handles abbreviations: `(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])`

## Adding New Course Documents

Place `.txt` files in `/docs` folder. On startup, `app.py:startup_event()` automatically:
1. Scans `/docs` folder
2. Processes each document
3. Adds to ChromaDB (skips duplicates by course title)

Alternatively, use RAGSystem methods programmatically:
```python
rag_system.add_course_document(file_path)           # Single file
rag_system.add_course_folder(folder_path, clear_existing=False)  # Batch
```

## Extending the System

### Adding New Tools
1. Create class implementing `Tool` abstract base class
2. Implement `get_tool_definition()` - return Anthropic tool schema
3. Implement `execute(**kwargs)` - tool logic
4. Register with ToolManager: `tool_manager.register_tool(my_tool)`

### Modifying Search Behavior
Edit `AIGenerator.SYSTEM_PROMPT` in `backend/ai_generator.py` to change how Claude uses tools.

### Changing Models
Update `config.py`:
- `ANTHROPIC_MODEL` - different Claude version
- `EMBEDDING_MODEL` - different Sentence-Transformer model (requires reprocessing documents)

## Frontend

Simple vanilla JavaScript SPA (`frontend/`):
- `script.js` - handles API calls, markdown rendering, session management
- `index.html` - chat interface + sidebar with course stats
- `style.css` - dark theme

Frontend uses Fetch API to communicate with backend. All state (sessions, history) managed server-side.

## Important Notes

- **First run** takes longer as ChromaDB initializes and documents are processed
- **ChromaDB data** persists in `./chroma_db` - delete to reprocess all documents
- **API calls** happen twice per query when tool is used (normal behavior)
- **Session IDs** are UUIDs - frontend tracks current session across requests
- **Tool execution** is synchronous - search completes before Claude sees results