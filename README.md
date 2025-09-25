# SOPS-AI MCP Server

FastMCP server that exposes three powerful tools to Model Context Protocol (MCP) compatible clients:

1. `web_search` – Real‑time web search using the official Tavily Python SDK
2. `kb_search` – Semantic knowledge base retrieval via Pinecone + OpenAI embeddings
3. `create_request` – Create external service / ticketing requests through a REST API

> Built with FastMCP for quick authoring, strong typing, and simple deployment.

## ✨ Features

- Tavily web search (answer synthesis, domain include/exclude, images, raw content)
- Vector similarity search over your Pinecone index (OpenAI `text-embedding-3-small`)
- External Request / Ticket creation with validation & detailed error handling
- Environment‑driven configuration (12‑factor friendly)
- Clear separation of tool logic for reuse & direct testing (`test/test-mcp.py`)

## 🧱 Architecture

```
FastMCP (server runtime)
├── web_search  -> Tavily SDK
├── kb_search   -> OpenAI Embeddings -> Pinecone Query
└── create_request -> Signed REST POST to external Request API
```

All tools are registered via FastMCP decorators in `src/main.py`. The same underlying logic is exercised directly by the test scripts without spinning up an MCP runtime.

## 📦 Requirements

| Component | Purpose |
|-----------|---------|
| Python 3.12+ | Runtime |
| fastmcp | MCP server framework |
| tavily-python | Web search SDK |
| openai | Embedding generation |
| pinecone-client | Vector DB access |
| requests | External API integration |
| python-dotenv | Local `.env` loading |

## 🔐 Environment Variables

Create a `.env` file (or export these in your shell):

```
TAVILY_API_KEY=your_tavily_key
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
REQUEST_SERVER_URL=https://request.example.com
REQUEST_ACCESS_TOKEN=your_request_api_bearer_token
```

Optional: only the tools whose dependencies are fully configured will be functional; others return a friendly error block.

## 🚀 Installation & Run

```bash
git clone <repo-url>
cd mcp-server
python -m venv .venv
source .venv/bin/activate  # zsh/bash
pip install --upgrade pip
pip install -e .            # uses pyproject dependencies

# or explicit
# pip install fastmcp tavily-python openai pinecone-client python-dotenv requests

cp .env.example .env  # (create if you provide a sample)
python src/main.py
```

### 🔄 Using uv (fast installer & resolver)

If you prefer [uv](https://github.com/astral-sh/uv) for speed & reproducibility:

```bash
# 1. Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install uv

# 2. Create & sync the virtual environment from pyproject.toml
uv sync          # creates .venv/ automatically and installs deps

# 3. Run the server inside the managed environment
uv run python src/main.py

# (Optional) Add a new dependency (writes to pyproject.toml and locks versions)
uv add rich

# Upgrade all dependencies against latest compatible versions
uv lock --upgrade

# Remove unused/downloaded wheel cache (housekeeping)
uv cache prune
```

Notes:
- Commit the generated `uv.lock` (if you run `uv lock`) for deterministic builds.
- `uv run` isolates execution without needing to manually activate `.venv`.
- Use `uv pip sync pyproject.toml` if you want to force exact lock state into an already existing environment.

You should see startup diagnostics showing which integrations are active.

## 🛠 Tools Overview

### 1. web_search
Inputs:
- `query` (str, required)
- `max_results` (int, default 5)
- `search_depth` ("basic" | "advanced")
- `include_answer`, `include_raw_content`, `include_images` (bool flags)
- `include_domains`, `exclude_domains` (list[str])

Returns synthesized answer + structured results.

### 2. kb_search
Performs: OpenAI embedding -> Pinecone similarity query.

Inputs: `query` (str), `top_k` (int), `include_metadata` (bool)

Response includes scored matches and metadata (index name, model).

### 3. create_request
Validates + POSTs a ticket to the external Request API with detailed status‑aware error mapping.

Minimum required: `subject`, `requester_email`.

Optional rich fields: category, impact/priority/urgency, support level (`tier1..tier4`), tags, department, location, assignee, technician group, CC set, linkage arrays, custom fields, attachments.

## 🧪 Testing

Direct (no MCP runtime) functional checks:
```bash
python test/test-mcp.py
```

This script:
- Verifies env var presence
- Exercises web, KB, and request creation flows (minimal + full)
- Prints structured success / error diagnostics

Tavily quick probe (ensure you DO NOT hardcode keys in committed code):
```bash
python test/test-tavily.py  # adapt to load key from env before committing
```

> Note: remove any hardcoded API key before publishing. Use `os.getenv("TAVILY_API_KEY")` instead.

## 🧪 Example MCP Client Invocation (Conceptual)

If your MCP client supports JSON tool calls, a `web_search` invocation might look like:
```jsonc
{
	"tool": "web_search",
	"args": { "query": "Latest FastMCP examples", "max_results": 3 }
}
```

## ❗️ Error Handling Philosophy

- Missing integration → returns `{ "error": "... not initialized" }`
- External HTTP failures → categorized (400, 401, 403, 404, 500, fallback)
- Validation guards for enum & email formats

## 🔍 Observability

Current version logs startup capability matrix to stdout. For production consider:
- Structured logging (json)
- Request/response latency metrics
- Retry/backoff wrappers for transient network errors

## 🧩 Extending

Add a new tool:
```python
@mcp.tool()
def my_new_tool(arg1: str) -> dict:
		# implement
		return {"ok": True}
```
Restart the server; FastMCP auto‑registers the function.

## 🔒 Security Notes

- Never commit secrets; prefer `.env` + secret manager in production.
- Validate all external inputs (already partially done for emails / enums).
- Consider rate limiting if publicly exposed.

## 🗺 Roadmap Ideas

- Async variants (reduce latency via parallel I/O)
- Caching layer for repeated Tavily queries
- Batch embedding + ingestion script for Pinecone
- Health endpoint & OpenTelemetry traces
- Better schema / Pydantic models for tool IO

## 🤝 Contributing

1. Fork & branch
2. Add feature / test
3. Run tests & lint
4. Open PR describing change and integration impacts

## 📄 License

Add a license file (MIT / Apache-2.0 recommended) and reference it here.

## ✅ Quick Start Recap

```bash
pip install -e .
cp .env.example .env   # fill keys
python src/main.py
```

You now have an MCP server exposing search, knowledge, and ticketing automation tools. Plug it into your MCP aware client and build higher‑level workflows fast.

---
Generated README – adjust branding, roadmap, and license to match your org before publishing.

