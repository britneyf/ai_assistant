"""
Audit Assistant API - FastAPI server for audit information queries and document analysis.

Features:
- Query existing audit information from Excel context files (audits, issues, workpapers)
- Upload and analyze documents (PDF, DOCX, TXT) with AI-powered summaries and Q&A
- Chat interface supporting both Excel context queries and document Q&A
- Uses LangChain and OpenAI for intelligent analysis
"""

import os
import uuid
import traceback
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict

from dotenv import load_dotenv

import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory for uploaded documents
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

# Directory for Excel context files (reference data)
CONTEXT_DIR = Path(os.getenv("CONTEXT_DIR", "test_files"))
CONTEXT_DIR.mkdir(exist_ok=True)

# Vector store directory for document embeddings
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "vector_stores"))
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# File upload settings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
CHUNK_SIZE = 1024 * 1024  # 1 MB chunks for streaming upload

# Supported document file types
SUPPORTED_DOC_TYPES = {".pdf", ".docx", ".txt", ".doc"}

# Serve frontend only in testing/development mode
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "false").lower() in ("true", "1", "yes")

# OpenAI configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# ============================================================================
# GLOBAL STATE - Excel Context Data (loaded at startup)
# ============================================================================

# Store Excel context dataframes in memory for fast querying
EXCEL_CONTEXT: Dict[str, pd.DataFrame] = {}

# Store document vector stores (one per uploaded document)
DOCUMENT_STORES: Dict[str, Any] = {}

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="Audit Assistant API",
    description="AI-powered assistant for querying audit information and analyzing documents",
    version="1.0.0"
)

# CORS middleware - allows frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class ChatMessage(BaseModel):
    """Single chat message model."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint - Mastra-enabled with runtime context."""
    message: str
    conversation_id: Optional[str] = None  # For maintaining conversation history
    context_type: Optional[str] = None  # "excel" or "document" or None (auto-detect)
    # Mastra runtime context support
    resource_id: Optional[str] = None  # e.g., "audit-123" - Mastra resource identifier
    thread_id: Optional[str] = None  # e.g., "workpaper-session" - Mastra thread identifier
    metadata: Optional[Dict[str, Any]] = None  # Additional Mastra context metadata


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    conversation_id: str
    context_used: str  # "excel" or "document" or "both"
    sources: Optional[List[str]] = None  # For document Q&A, list source documents


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    original_name: str
    size_bytes: int
    size_mb: float
    upload_time: str
    summary: Optional[str] = None
    file_type: str


class FileInfo(BaseModel):
    """File information response model."""
    filename: str
    original_name: str
    size_bytes: int
    size_mb: float
    upload_time: str
    file_type: str

# ============================================================================
# EXCEL CONTEXT LOADING FUNCTIONS
# ============================================================================

def load_excel_file(filepath: Path, header_row: int = 4) -> pd.DataFrame:
    """
    Load an Excel file with proper header detection.
    
    Args:
        filepath: Path to the Excel file
        header_row: Row index where headers are located (default: 4 for audit reports)
    
    Returns:
        DataFrame with properly parsed headers
    """
    try:
        df = pd.read_excel(filepath, header=header_row)
        # Clean up: remove rows where all values are NaN
        df = df.dropna(how='all')
        return df
    except Exception as e:
        raise ValueError(f"Error loading Excel file {filepath}: {str(e)}")


def load_excel_context(context_dir: Path = CONTEXT_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load all Excel context files at startup.
    
    These files provide the structure and reference data for the audit system:
    - list_of_audits.xlsx: Audit metadata (titles, IDs, managers, status, dates)
    - list_of_issues.xlsx: Issues/findings from audits (titles, ratings, status, owners)
    - workpapers_by_status.xlsx: Workpaper information (titles, status, auditors, dates)
    
    Args:
        context_dir: Directory containing Excel context files
    
    Returns:
        Dictionary mapping context names to DataFrames
    """
    context = {}
    
    # Define the context files we expect
    context_files = {
        "audits": "list_of_audits.xlsx",
        "issues": "list_of_issues.xlsx",
        "workpapers": "workpapers_by_status.xlsx"
    }
    
    for context_name, filename in context_files.items():
        filepath = context_dir / filename
        if filepath.exists():
            try:
                df = load_excel_file(filepath)
                context[context_name] = df
                print(f"✓ Loaded {context_name} context: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"⚠ Warning: Could not load {filename}: {str(e)}")
        else:
            print(f"⚠ Warning: Context file not found: {filename}")
    
    return context


def get_excel_context_summary() -> str:
    """
    Get a comprehensive summary of the loaded Excel context for the LLM.
    
    This provides domain knowledge about the audit system structure, terminology,
    and relationships to help the assistant understand the audit domain better.
    
    Returns:
        String description of audit system context
    """
    if not EXCEL_CONTEXT:
        return "No Excel context data loaded."
    
    summary_parts = []
    summary_parts.append("AUDIT SYSTEM CONTEXT (for understanding audit domain):")
    summary_parts.append("=" * 80)
    
    # Define the actual column structures as provided
    column_structures = {
        "audits": [
            "Audit Title", "Audit Group", "Plan Title", "Start Date", "End Date", 
            "Status", "Classification", "Type", "Scope Elements", "Created By", 
            "Issues", "Workpapers", "Audit Rating", "Report Transactions", 
            "Audit Closure", "Audit ID", "Audit Plan ID", "Audit Manager", "Lead Auditor"
        ],
        "issues": [
            "Audit Title", "Issue Title", "Workpaper Title", "Issue Owner", 
            "Issue Rating", "Priority", "Issue Due Date", "Issue Created By", 
            "Issue Disposition", "Issue Status", "Issue Progress", "No.of Actions", 
            "Issue Approvers", "Approver Roles", "Approval", "Related Objects", 
            "Issue Types", "Repeat Issue", "Related Issue", "Issue ID", 
            "Audit Issue ID", "Current Assignee"
        ],
        "workpapers": [
            "Audit Title", "Workpaper Title", "Status", "Auditor", "Approver", 
            "Current Assignee", "Start Date", "End Date", "Audit Title"
        ]
    }
    
    # Relationship mapping - based on actual column structures
    relationships = """
KEY RELATIONSHIPS BETWEEN DATA SOURCES:
- PRIMARY KEY: All three sources share "Audit Title" - use this to join/relate data
- Audits table: Contains audit metadata (titles, dates, status, managers, IDs)
- Issues table: Has "Audit Title" (links to audits) and "Workpaper Title" (links to workpapers)
- Workpapers table: Has "Audit Title" (links to audits)
- To find issues for an audit: Filter issues where "Audit Title" matches
- To find workpapers for an audit: Filter workpapers where "Audit Title" matches  
- To find issues for a workpaper: Match "Workpaper Title" between issues and workpapers
- When querying across sources, use pandas merge/join on "Audit Title"
"""
    
    for name, df in EXCEL_CONTEXT.items():
        summary_parts.append(f"\n{name.upper().replace('_', ' ')}:")
        summary_parts.append(f"  - Total records: {len(df)}")
        
        # Use actual column structure if available, otherwise use what's in the dataframe
        expected_columns = column_structures.get(name, [])
        actual_columns = df.columns.tolist()
        
        # Try to match actual columns to expected structure
        summary_parts.append(f"  - Key columns: {', '.join(actual_columns[:20])}")
        
        # Show sample data
        if len(df) > 0:
            sample_size = min(3, len(df))
            sample_data = df.head(sample_size)
            summary_parts.append(f"  - Sample data (first {sample_size} rows):")
            for idx, row in sample_data.iterrows():
                row_summary = []
                for col in actual_columns[:5]:  # Show first 5 columns
                    val = row[col]
                    if pd.notna(val):
                        val_str = str(val)[:50]
                        row_summary.append(f"{col}: {val_str}")
                summary_parts.append(f"    Row {idx+1}: {' | '.join(row_summary)}")
    
    summary_parts.append("\n" + relationships)
    summary_parts.append("\n" + "=" * 80)
    summary_parts.append("Use this context to understand audit terminology, relationships,")
    summary_parts.append("and structure when answering questions about uploaded documents.")
    
    return "\n".join(summary_parts)

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def extract_text_from_document(filepath: Path, file_type: str) -> str:
    """
    Extract text content from a document file.
    
    Supports:
    - PDF files (.pdf)
    - Word documents (.docx, .doc)
    - Text files (.txt)
    
    Args:
        filepath: Path to the document file
        file_type: File extension (e.g., ".pdf", ".docx")
    
    Returns:
        Extracted text content
    """
    file_type_lower = file_type.lower()
    
    try:
        if file_type_lower == ".pdf":
            # Use PyPDFLoader from LangChain
            loader = PyPDFLoader(str(filepath))
            documents = loader.load()
            return "\n\n".join([doc.page_content for doc in documents])
        
        elif file_type_lower in [".docx", ".doc"]:
            # Use Docx2txtLoader from LangChain
            loader = Docx2txtLoader(str(filepath))
            documents = loader.load()
            return "\n\n".join([doc.page_content for doc in documents])
        
        elif file_type_lower == ".txt":
            # Use TextLoader from LangChain
            loader = TextLoader(str(filepath))
            documents = loader.load()
            return "\n\n".join([doc.page_content for doc in documents])
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    except Exception as e:
        raise ValueError(f"Error extracting text from document: {str(e)}")


def create_document_vector_store(document_id: str, text: str) -> Any:
    """
    Create a vector store for a document to enable semantic search and Q&A.
    
    Uses ChromaDB for vector storage and OpenAI embeddings for text embeddings.
    
    Args:
        document_id: Unique identifier for the document
        text: Text content of the document
    
    Returns:
        Chroma vector store object
    """
    # Split text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    persist_directory = str(VECTOR_STORE_DIR / document_id)
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_store


def generate_document_summary(text: str, max_length: int = 500) -> str:
    """
    Generate a summary of a document using OpenAI.
    
    Args:
        text: Document text content
        max_length: Maximum length of summary in characters
    
    Returns:
        Document summary
    """
    # Truncate text if too long (to save tokens)
    if len(text) > 10000:
        text = text[:10000] + "..."
    
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    
    prompt = f"""Please provide a concise summary of the following document. 
    Focus on key points, main topics, and important information.
    Keep the summary under {max_length} characters.
    
    Document content:
    {text}
    
    Summary:"""
    
    try:
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        return summary[:max_length]
    except Exception as e:
        return f"Summary generation failed: {str(e)}"

# ============================================================================
# QUERY ROUTING AND VALIDATION
# ============================================================================

def detect_query_type(query: str) -> str:
    """
    Detect whether a query is about Excel context data or uploaded documents.
    
    Args:
        query: User's query message
    
    Returns:
        "excel" if query is about audit data from Excel files
        "document" if query is about uploaded documents
        "both" if unclear (will try both)
    """
    query_lower = query.lower()
    
    # Keywords that suggest Excel context queries (about existing audit data)
    excel_keywords = [
        "audit", "audits", "issue", "issues", "workpaper", "workpapers",
        "audit manager", "lead auditor", "audit status", "audit rating",
        "finding", "findings", "disposition", "approver", "how many",
        "list", "show me", "what are", "which audits", "pending"
    ]
    
    # Keywords that suggest document queries
    document_keywords = [
        "document", "uploaded", "file", "summary", "what does the document say",
        "in the document", "from the document", "the uploaded"
    ]
    
    excel_score = sum(1 for keyword in excel_keywords if keyword in query_lower)
    document_score = sum(1 for keyword in document_keywords if keyword in query_lower)
    
    # If explicitly about documents, use document mode
    if document_score > 0 and "uploaded" in query_lower or "document" in query_lower:
        return "document"
    # If about audit data and no document keywords, use Excel
    elif excel_score > 0 and document_score == 0:
        return "excel"
    # Default to trying both
    else:
        return "both"


def validate_query_relevance(query: str) -> Tuple[bool, str]:
    """
    Validate if the query is relevant to audit/GRC topics.
    
    Uses simple keyword matching first, then AI validation if needed.
    This prevents false negatives for clearly audit-related queries.
    
    Args:
        query: User's query message
    
    Returns:
        Tuple of (is_relevant, denial_message)
    """
    query_lower = query.lower()
    
    # Quick keyword check - if query contains audit-related keywords, allow it immediately
    audit_keywords = [
        "audit", "audits", "issue", "issues", "workpaper", "workpapers",
        "finding", "findings", "compliance", "risk", "governance",
        "control", "controls", "document", "documents", "uploaded",
        "2025", "2024", "2023", "date", "status", "manager", "auditor"
    ]
    
    # If query contains any audit keywords, allow it
    if any(keyword in query_lower for keyword in audit_keywords):
        return True, ""
    
    # For queries without clear keywords, use AI validation
    try:
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.1,
            timeout=10,
            max_retries=1,
            request_timeout=10
        )
        
        validation_prompt = f"""You are a query validator for an Audit and GRC (Governance, Risk, Compliance) assistant.

The user asked: "{query}"

Determine if this query is relevant to:
1. Audit topics: audits, audit management, audit findings, issues, workpapers, audit reports, compliance, controls, risk management
2. GRC topics: governance, risk management, compliance, controls, policies, regulations, security, frameworks
3. Document analysis: questions about uploaded documents, document summaries, document content

Consider the query relevant if it:
- Asks about audits, issues, workpapers, or audit data
- Asks about GRC topics (risk, compliance, governance)
- Asks about analyzing or querying uploaded documents
- Seeks insights from audit data or documents
- Mentions years, dates, or time periods in context of audits/data

Consider the query irrelevant if it:
- Asks about unrelated topics (weather, recipes, entertainment, news, sports)
- Requests general knowledge not related to audits/GRC
- Is casual conversation unrelated to business/audit topics

IMPORTANT: Be permissive - if there's any chance the query is about audits, data, or documents, mark it as relevant.

Respond with ONLY "YES" if relevant, or "NO" if irrelevant.

Response:"""
        
        response = llm.invoke(validation_prompt)
        response_text = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
        
        if "NO" in response_text and "NOT RELEVANT" in response_text and "IRRELEVANT" in response_text:
            # Only reject if explicitly and clearly irrelevant
            return False, "I can only answer questions related to audits, GRC (Governance, Risk, Compliance), and document analysis. Please ask questions about audit data, compliance, risk management, or uploaded documents."
        
        # Default to allowing if unclear
        return True, ""
    except Exception as e:
        # If validation fails, default to allowing (to avoid blocking valid queries)
        print(f"Warning: Query validation failed: {str(e)}. Allowing query to proceed.")
        return True, ""

# ============================================================================
# EXCEL CONTEXT QUERY FUNCTIONS
# ============================================================================

def query_excel_context(query: str) -> Dict[str, Any]:
    """
    Query the Excel context data using LangChain pandas agent.
    
    This function allows natural language queries against the audit, issues, and workpapers data.
    Uses the same approach as the reference implementation for consistent formatting.
    
    Args:
        query: Natural language query about audit data
    
    Returns:
        Dictionary with insights, business_insights, and success status
    """
    if not EXCEL_CONTEXT:
        return {
            "insights": "No Excel context data loaded. Please ensure context files are available.",
            "business_insights": "",
            "success": False
        }
    
    # Combine all context dataframes for comprehensive querying
    # Add a source column to identify which context each row came from
    combined_dataframes = []
    file_info = []
    for context_name, df in EXCEL_CONTEXT.items():
        df_copy = df.copy()
        df_copy['_context_source'] = context_name
        combined_dataframes.append(df_copy)
        file_info.append({
            "filename": context_name,
            "total_rows": len(df),
            "total_rows_in_file": len(df),
            "is_sample": False,
            "columns": list(df.columns),
        })
    
    # Combine all dataframes
    if len(combined_dataframes) == 1:
        main_df = combined_dataframes[0]
    else:
        main_df = pd.concat(combined_dataframes, ignore_index=True, sort=False)
    
    # Use the same analyze_with_langchain approach from reference
    # Import the function logic inline
    import json
    import re
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.1,
        timeout=120,
        max_retries=2,
        request_timeout=120
    )
    
    # Build metadata text similar to reference implementation
    files_metadata_text = f"\n\n{'='*80}\nAUDIT DATA CONTEXT\n{'='*80}\n"
    files_metadata_text += f"You are working with {len(combined_dataframes)} data sources. Each source's structure is detailed below:\n\n"
    
    for i, (df, info) in enumerate(zip(combined_dataframes, file_info), 1):
        filename = info.get('filename', f'source_{i}')
        total_rows = info.get('total_rows_in_file', len(df))
        loaded_rows = len(df)
        columns = list(df.columns)
        
        # Get first 5 rows as sample
        sample_rows = df.head(5).to_dict(orient='records')
        
        files_metadata_text += f"\n{'─'*80}\n"
        files_metadata_text += f"SOURCE {i}: {filename}\n"
        files_metadata_text += f"{'─'*80}\n"
        files_metadata_text += f"Total Rows: {total_rows:,} (Loaded: {loaded_rows:,})\n"
        files_metadata_text += f"Columns ({len(columns)}): {', '.join(columns)}\n"
        files_metadata_text += f"\nColumn Data Types:\n"
        for col in columns:
            dtype = str(df[col].dtype)
            files_metadata_text += f"  - {col}: {dtype}\n"
        
        files_metadata_text += f"\nSample Data (First 5 rows):\n"
        for row_idx, row in enumerate(sample_rows, 1):
            files_metadata_text += f"\n  Row {row_idx}:\n"
            for col, val in row.items():
                val_str = str(val)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                files_metadata_text += f"    {col}: {val_str}\n"
        
        files_metadata_text += "\n"
    
    files_metadata_text += f"\n{'='*80}\n"
    files_metadata_text += "KEY RELATIONSHIPS FOR JOINING DATA:\n"
    files_metadata_text += "- PRIMARY KEY: All three sources share 'Audit Title' - use this to join/relate data across sources\n"
    files_metadata_text += "\nCOLUMN STRUCTURES:\n"
    files_metadata_text += "- Audits: Audit Title, Audit Group, Plan Title, Start Date, End Date, Status, Classification, Type, Scope Elements, Created By, Issues, Workpapers, Audit Rating, Report Transactions, Audit Closure, Audit ID, Audit Plan ID, Audit Manager, Lead Auditor\n"
    files_metadata_text += "- Issues: Audit Title, Issue Title, Workpaper Title, Issue Owner, Issue Rating, Priority, Issue Due Date, Issue Created By, Issue Disposition, Issue Status, Issue Progress, No.of Actions, Issue Approvers, Approver Roles, Approval, Related Objects, Issue Types, Repeat Issue, Related Issue, Issue ID, Audit Issue ID, Current Assignee\n"
    files_metadata_text += "- Workpapers: Audit Title, Workpaper Title, Status, Auditor, Approver, Current Assignee, Start Date, End Date\n"
    files_metadata_text += "\nJOINING INSTRUCTIONS:\n"
    files_metadata_text += "- To find issues for a specific audit: filter issues where 'Audit Title' matches the audit\n"
    files_metadata_text += "- To find workpapers for a specific audit: filter workpapers where 'Audit Title' matches the audit\n"
    files_metadata_text += "- To find issues for a workpaper: use 'Workpaper Title' in issues table to match 'Workpaper Title' in workpapers\n"
    files_metadata_text += "- Issues table has both 'Audit Title' and 'Workpaper Title' - can link to both audits and workpapers\n"
    files_metadata_text += "- When querying across sources, use pandas merge/join operations on 'Audit Title'\n"
    files_metadata_text += "- Example: To get all issues for audits in 2025, first filter audits by Start Date year, then join with issues on 'Audit Title'\n"
    files_metadata_text += "- The '_context_source' column indicates which source (audits, issues, or workpapers) each row came from\n"
    files_metadata_text += f"{'='*80}\n"
    
    # Enhanced query with same format as reference implementation
    enhanced_query = f"""{query}{files_metadata_text}

COMBINED DATASET:
- All {len(combined_dataframes)} sources have been COMBINED into a single dataset
- The combined dataset contains {len(main_df):,} rows from all sources
- A '_context_source' column identifies which source each row came from
- Use the metadata above to understand relationships and perform intelligent queries

CONTEXT AWARENESS:
- You are analyzing audit and risk-related data
- When interpreting data, consider risk implications (e.g., audit findings relate to risk, issues indicate risk areas)
- If the query relates to risk, compliance, controls, or audit findings, prioritize that perspective
- Connect data patterns to potential risk indicators where relevant

CRITICAL INSTRUCTIONS (BE EFFICIENT - MINIMIZE STEPS):
1. Execute the query DIRECTLY - use simple pandas operations (head, value_counts, groupby, etc.)
2. DO NOT iterate unnecessarily - get the answer in 1-3 steps maximum
3. If asking for a summary/list, use .head(10), .value_counts(), or .describe() directly
4. Return your response as a JSON object with the following structure

MANDATORY RESPONSE FORMAT (JSON):
{{
  "message": "Direct answer to the user's question with actual numbers and names (not IDs) from your analysis. This should be a clear, concise response that directly addresses what was asked.",
  "business_insights": "OPTIONAL - Only include if the user explicitly asks for insights, analysis, implications, or recommendations. For simple factual questions (like 'what are audits from 2025'), leave this as an empty string.",
  "math_insights": {{
    "metrics": [
      {{"label": "Metric Name", "value": "actual_number", "trend": "up|down|neutral", "interpretation": "What this metric means for the business"}}
    ],
    "summary": "Technical summary with real numbers"
  }},
  "chart": null,
  "followup_question": ""
}}

RESPONSE GUIDELINES:
- The "message" field should directly answer the user's question in a natural, conversational way
- Use plain text ONLY - NO markdown formatting (no **bold**, no numbered lists with "1.", "2.", "3.", no bullet points, no asterisks)
- Write in a natural, conversational style - like you're speaking to someone directly
- For simple factual questions (e.g., "what are audits from 2025", "show me all audits", "how many audits"), provide a clear direct answer
- Instead of numbered lists or bullet points, use simple flowing sentences separated by periods or commas
- Example GOOD format: "In 2025, three audits took place. AML Audit started on December 4, AML and KYC Audit started on October 31, and AML and kYC Audit started on October 27. All three are currently in 'Audit Started' status."
- Example BAD format: "1. **AML Audit** - Started on December 4\n2. **AML and KYC Audit** - Started on October 31" (DO NOT use this format)
- When listing multiple items, use natural language like "There are X audits: [item1], [item2], and [item3]." or separate with periods
- Only include "business_insights" if the user asks for analysis, insights, implications, trends, or recommendations
- If the user just wants facts/data, set "business_insights" to an empty string ""
- Keep responses concise and focused on what was asked

DATE FILTERING:
- When filtering by year (e.g., "2025", "from 2025", "in 2025"), you MUST:
  1. Identify date columns (columns with "date" in the name, especially "Start Date", "End Date")
  2. Extract the year from date columns using: df[date_col].dt.year if datetime, or pd.to_datetime(df[date_col]).dt.year if string
  3. Filter using: df[df[date_col].dt.year == 2025] for datetime columns
  4. For queries like "from 2025" or "in 2025", filter where the year equals 2025
  5. If the date column is a string, convert it first: pd.to_datetime(df[date_col], errors='coerce')
  6. Check BOTH start date and end date columns if both exist, unless the query specifically mentions one
- ALWAYS verify your date filtering is working by checking: df[date_col].dt.year.unique() or similar

RESPONSE STYLE:
- Answer directly and naturally - like you're having a conversation
- Use plain, flowing sentences - avoid structured formats like numbered lists or bullet points
- When listing items, write them naturally in sentences, not as formatted lists
- Avoid mentioning technical terms like "dataframe", "column", "df", "pandas", etc.
- Execute the query efficiently - use direct pandas operations
- Return ONLY the final results - no intermediate steps needed
- For simple questions, just answer the question without extra analysis

EFFICIENCY TIPS:
- For "show me" or "list" queries: Use .head(10) or .to_dict(orient='records')[:10]
- For counts: Use len(df) or df['column'].value_counts()
- For summaries: Use df.describe() or df.groupby().agg()
- For filtering: Use df[df['column'] == 'value'] directly
- DO NOT loop through rows manually - use vectorized operations

IMPORTANT: Your response MUST be valid JSON. Start with {{ and end with }}. Do not include markdown code blocks. Be direct and efficient - answer the question in minimal steps. For simple factual questions, leave business_insights as an empty string.
"""
    
    # Create pandas agent
    try:
        max_iterations = int(os.getenv("MAX_AGENT_ITERATIONS", "25"))
        
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=main_df,
            verbose=False,
            allow_dangerous_code=True,
            agent_type="openai-functions",
            max_iterations=max_iterations,
            max_execution_time=180
        )
        
        # Run agent with timeout
        analysis_timeout = int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "180"))
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.invoke, {"input": enhanced_query})
            try:
                result = future.result(timeout=analysis_timeout)
            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError(f"Analysis exceeded maximum time limit of {analysis_timeout} seconds")
        
        # Extract output
        if isinstance(result, dict):
            raw_output = result.get("output", result.get("result", str(result)))
        else:
            raw_output = str(result)
        
        # Try to parse JSON from the response
        insights_text = raw_output
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = None
        
        if json_str:
            try:
                parsed_response = json.loads(json_str)
                # Extract components
                insights = parsed_response.get("message", insights_text)
                business_insights = parsed_response.get("business_insights", "")
                if not business_insights or not business_insights.strip():
                    business_insights = ""
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output
                insights = insights_text
                business_insights = ""
        else:
            # No JSON found, return raw output
            insights = insights_text
            business_insights = ""
        
        return {
            "insights": insights,
            "business_insights": business_insights,
            "success": True
        }
    
    except TimeoutError as e:
        return {
            "insights": f"Analysis timed out. The query may be too complex. Please try a simpler query.",
            "business_insights": "",
            "success": False
        }
    except Exception as e:
        error_msg = str(e)
        if "max_iterations" in error_msg.lower() or "too many iterations" in error_msg.lower():
            insights = f"Analysis exceeded maximum iterations. The query may be too complex. Please try a simpler or more specific query."
        else:
            insights = f"Error querying audit context: {error_msg}"
        return {
            "insights": insights,
            "business_insights": "",
            "success": False
        }

# ============================================================================
# DOCUMENT Q&A FUNCTIONS
# ============================================================================

def query_documents(query: str, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Query uploaded documents using RAG (Retrieval Augmented Generation).
    
    This function uses the Excel context to provide domain knowledge about audits,
    helping the assistant understand audit terminology and structure when answering
    questions about documents.
    
    Args:
        query: User's question about the documents
        document_ids: Specific document IDs to query (None = query all documents)
    
    Returns:
        Dictionary with answer, sources, and success status
    """
    if not DOCUMENT_STORES:
        return {
            "answer": "No documents have been uploaded yet. Please upload documents first.",
            "sources": [],
            "success": False
        }
    
    # Determine which documents to query
    if document_ids:
        stores_to_query = {doc_id: DOCUMENT_STORES[doc_id] for doc_id in document_ids if doc_id in DOCUMENT_STORES}
    else:
        stores_to_query = DOCUMENT_STORES
    
    if not stores_to_query:
        return {
            "answer": "No matching documents found.",
            "sources": [],
            "success": False
        }
    
    # Get Excel context to inform the assistant's understanding
    audit_context = get_excel_context_summary()
    
    # Initialize LLM
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    
    # Query each document store and combine results
    all_answers = []
    all_sources = []
    
    for doc_id, vector_store in stores_to_query.items():
        try:
            # Create retrieval QA chain with custom prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            # Enhance query with audit context to help the LLM understand domain terminology
            # The audit context provides background knowledge about audit structure
            enhanced_query = f"""You are an expert audit and GRC (Governance, Risk, Compliance) assistant.

AUDIT SYSTEM CONTEXT (for understanding audit terminology and structure):
{audit_context[:2000]}

USER QUESTION: {query}

Answer the question based on the document content. Use audit terminology appropriately based on the context above.
If the answer is not in the documents, say so."""
            
            result = qa_chain.invoke({"query": enhanced_query})
            
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            sources = [doc_id] if source_docs else []
            
            if answer:
                all_answers.append(answer)
                all_sources.extend(sources)
        
        except Exception as e:
            print(f"Error querying document {doc_id}: {str(e)}")
            # Fallback to simple query without custom prompt
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                result = qa_chain.invoke({"query": query})
                answer = result.get("result", "")
                if answer:
                    all_answers.append(answer)
                    all_sources.append(doc_id)
            except:
                pass
    
    # Combine answers
    if not all_answers:
        return {
            "answer": "No answer found in the documents.",
            "sources": [],
            "success": False
        }
    
    combined_answer = "\n\n".join(all_answers) if len(all_answers) > 1 else all_answers[0]
    
    return {
        "answer": combined_answer,
        "sources": list(set(all_sources)),
        "success": True
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load Excel context files and initialize agent system when the server starts."""
    global EXCEL_CONTEXT
    print("Loading Excel context files...")
    EXCEL_CONTEXT = load_excel_context()
    print(f"✓ Loaded {len(EXCEL_CONTEXT)} context files")
    
    # Initialize agent system
    from agents import initialize_agent_system
    initialize_agent_system(EXCEL_CONTEXT, DOCUMENT_STORES, OPENAI_MODEL, get_excel_context_summary)
    print("✓ Agent system initialized")


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint - serves frontend if enabled, otherwise health check."""
    if SERVE_FRONTEND:
        index_path = Path("index.html")
        if index_path.exists():
            return FileResponse(index_path)
    return {
        "status": "healthy",
        "service": "Audit Assistant API",
        "version": "1.0.0",
        "context_loaded": len(EXCEL_CONTEXT) > 0
    }


@app.get("/frontend")
async def frontend():
    """Serve the frontend interface (only available when SERVE_FRONTEND=true)."""
    if not SERVE_FRONTEND:
        raise HTTPException(
            status_code=403,
            detail="Frontend is disabled. Set SERVE_FRONTEND=true to enable."
        )
    index_path = Path("index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Audit Assistant API",
        "version": "1.0.0",
        "context_files_loaded": len(EXCEL_CONTEXT),
        "documents_loaded": len(DOCUMENT_STORES)
    }


@app.get("/context/summary")
async def get_context_summary():
    """Get summary of loaded Excel context."""
    return {
        "context_files": list(EXCEL_CONTEXT.keys()),
        "summary": get_excel_context_summary(),
        "details": {
            name: {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist()
            }
            for name, df in EXCEL_CONTEXT.items()
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - uses agent system to handle queries.
    
    Routes queries to appropriate agents (Excel Query Agent or Document Q&A Agent).
    """
    # Import agent system (avoid circular import)
    from agents import agent_system
    
    if agent_system is None:
        # Fallback to old system if agents not initialized
        return ChatResponse(
            response="Agent system not initialized. Please restart the server.",
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            context_used="none",
            sources=None
        )
    
    # Validate query relevance
    is_relevant, denial_message = validate_query_relevance(request.message)
    if not is_relevant:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        return ChatResponse(
            response=denial_message,
            conversation_id=conversation_id,
            context_used="none",
            sources=None
        )
    
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Build context for agents with Mastra runtime context support
    context = {}
    if request.context_type:
        context["preferred_type"] = request.context_type
    
    # Add Mastra runtime context if provided
    if request.resource_id or request.thread_id:
        context["resource_id"] = request.resource_id
        context["thread_id"] = request.thread_id or conversation_id  # Use conversation_id as thread_id if not provided
        if request.metadata:
            context["metadata"] = request.metadata
    
    # Process query through agent system
    result = agent_system.process_query(request.message, context)
    
    # Map agent response to ChatResponse
    return ChatResponse(
        response=result.get("response", "No response generated."),
        conversation_id=conversation_id,
        context_used=result.get("context_used", result.get("agent", "unknown")),
        sources=result.get("sources", None)
    )


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT) for analysis.
    
    The document will be processed, embedded, and made available for Q&A.
    """
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_DOC_TYPES)}"
        )
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{document_id}_{file.filename}"
    filepath = UPLOAD_DIR / safe_filename
    
    # Stream file to disk with size validation
    total_size = 0
    try:
        with open(filepath, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    buffer.close()
                    filepath.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB"
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if filepath.exists():
            filepath.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )
    
    # Process document
    try:
        # Extract text
        text = extract_text_from_document(filepath, file_ext)
        
        # Generate summary
        summary = generate_document_summary(text)
        
        # Create vector store for Q&A
        vector_store = create_document_vector_store(document_id, text)
        DOCUMENT_STORES[document_id] = vector_store
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=safe_filename,
            original_name=file.filename,
            size_bytes=total_size,
            size_mb=round(total_size / (1024 * 1024), 2),
            upload_time=datetime.now().isoformat(),
            summary=summary,
            file_type=file_ext
        )
    
    except Exception as e:
        # Clean up on error
        if filepath.exists():
            filepath.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    documents = []
    for filepath in UPLOAD_DIR.glob("*"):
        if filepath.suffix.lower() in SUPPORTED_DOC_TYPES:
            stat = filepath.stat()
            documents.append({
                "filename": filepath.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_type": filepath.suffix.lower()
            })
    return {"documents": documents, "count": len(documents)}


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete an uploaded document and its vector store."""
    # Find and delete file
    deleted = False
    for filepath in UPLOAD_DIR.glob("*"):
        if document_id in filepath.name:
            filepath.unlink()
            deleted = True
            break
    
    # Delete vector store
    if document_id in DOCUMENT_STORES:
        del DOCUMENT_STORES[document_id]
    
    # Delete vector store directory
    vector_store_path = VECTOR_STORE_DIR / document_id
    if vector_store_path.exists():
        import shutil
        shutil.rmtree(vector_store_path, ignore_errors=True)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": f"Document {document_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
