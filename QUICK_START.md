# Quick Start - Mastra AI Assistant

## Project Structure

```
mastra_ai_assistant/
├── Python Backend (FastAPI)
│   ├── main.py              # FastAPI server
│   ├── agents.py            # Agent system with Mastra framework
│   ├── mastra_framework.py  # Mastra-inspired framework
│   └── requirements.txt     # Python dependencies
│
├── TypeScript Agent (Mastra)
│   ├── src/mastra/
│   │   ├── index.ts                    # Mastra configuration
│   │   ├── agents/
│   │   │   └── audit-chatbot-agent.ts  # Mastra agent
│   │   └── tools/
│   │       └── audit-chat-tool.ts      # Tool that calls Python API
│   └── package.json                    # Node.js dependencies
│
└── Data
    └── test_files/          # Excel context files (audits, issues, workpapers)
```

## How to Run

### Step 1: Start Python Backend

In one terminal:
```bash
cd /Users/bforsyth/Desktop/mastra_ai_assistant
python3 -m uvicorn main:app --reload
```

Backend will run at: `http://localhost:8000`

### Step 2: Start Mastra Dev Server

In another terminal:
```bash
cd /Users/bforsyth/Desktop/mastra_ai_assistant
npm run dev
```

This will:
- Start Mastra dev server
- Open Studio at `http://localhost:4111`
- Show the URL in terminal

### Step 3: Test in Mastra Studio

1. Open `http://localhost:4111` in your browser
2. Find the `audit-chatbot` agent
3. Try asking:
   - "How many audits are there?"
   - "What audits are currently in progress?"
   - "What issues are there?"

The agent will call your Python backend and return results!

## Environment Variables

The `.env` file contains:
- `OPENAI_API_KEY` - Your OpenAI API key
- `OPENAI_MODEL` - Model to use (gpt-4o-mini)
- `PYTHON_API_URL` - Python backend URL (http://localhost:8000)

## How It Works

1. **User asks question** in Mastra Studio
2. **Mastra agent** receives the question
3. **audit-chat-tool** calls Python API at `/chat`
4. **Python backend** processes using:
   - Pandas agents for Excel data
   - RAG for document Q&A
5. **Response** returns to Mastra Studio

## Deploy to Mastra Cloud

1. Deploy Python backend to Railway/Render/etc.
2. Set `PYTHON_API_URL` in Mastra Cloud environment variables
3. Run `npm run deploy` or deploy via Mastra Cloud

## Files

- **Python**: `main.py`, `agents.py`, `mastra_framework.py`
- **TypeScript**: `src/mastra/index.ts`, `src/mastra/agents/audit-chatbot-agent.ts`, `src/mastra/tools/audit-chat-tool.ts`
- **Config**: `package.json`, `.env`, `requirements.txt`

Everything is in one place: `mastra_ai_assistant`! 🎉
