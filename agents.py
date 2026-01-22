"""
Agent-based system for Audit Assistant - Mastra Framework Integration.

This module provides a modular agent architecture using Mastra-inspired patterns:
- Runtime context for dynamic agent behavior
- Memory/state management for conversation history
- Tools system for agent capabilities
- Workflow orchestration
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Import Mastra framework
from mastra_framework import (
    MastraAgent, RuntimeContext, MemoryStore, Tool, ToolRegistry, global_memory
)

# Import shared state and functions (will be set during initialization)
# These are imported at runtime to avoid circular imports
EXCEL_CONTEXT = None
DOCUMENT_STORES = None
OPENAI_MODEL = None
get_excel_context_summary = None


# ============================================================================
# BASE AGENT INTERFACE
# ============================================================================

class BaseAgent(ABC):
    """Base class for all agents in the audit assistant system."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def can_handle(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if this agent can handle the given query.
        
        Args:
            query: User's query
            context: Optional context information
            
        Returns:
            True if this agent can handle the query
        """
        pass
    
    @abstractmethod
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the agent's logic to process the query.
        
        Args:
            query: User's query
            context: Optional context information
            
        Returns:
            Dictionary with response, success status, and metadata
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata."""
        return {
            "name": self.name,
            "description": self.description
        }


# ============================================================================
# EXCEL QUERY AGENT
# ============================================================================

class ExcelQueryAgent(BaseAgent, MastraAgent):
    """Agent for querying Excel context data (audits, issues, workpapers) - Mastra-enabled."""
    
    def __init__(self):
        # Initialize BaseAgent for backward compatibility
        BaseAgent.__init__(
            self,
            name="excel_query",
            description="Queries audit data from Excel context files using natural language"
        )
        # Initialize MastraAgent with instructions and memory
        instructions = """You are an expert audit data analyst. Your role is to:
1. Query audit data from Excel context files (audits, issues, workpapers)
2. Provide clear, accurate answers based on the data
3. Use natural language to explain audit information
4. Reference specific audits, issues, or workpapers by name when possible
5. Include relevant dates, statuses, and ratings in your responses"""
        
        MastraAgent.__init__(
            self,
            name="excel_query",
            description="Queries audit data from Excel context files using natural language",
            instructions=instructions,
            memory=global_memory
        )
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.1,
            timeout=120,
            max_retries=2,
            request_timeout=120
        )
    
    def can_handle(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Detect if query is about Excel/audit data."""
        if not EXCEL_CONTEXT:
            return False
        
        # Keywords that suggest Excel/audit queries
        excel_keywords = [
            "audit", "issue", "workpaper", "status", "rating", "manager",
            "auditor", "priority", "disposition", "classification", "scope",
            "plan title", "audit group", "start date", "end date"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in excel_keywords)
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute Excel context query using LangChain pandas agent."""
        # Convert dict context to RuntimeContext if using Mastra
        mastra_context = None
        if context:
            mastra_context = RuntimeContext(
                resource_id=context.get("resource_id"),
                thread_id=context.get("thread_id"),
                metadata=context.get("metadata", {})
            )
        
        # Use Mastra generate method if context provided, otherwise use legacy method
        if mastra_context:
            return self._execute_mastra(query, mastra_context)
        else:
            return self._execute_legacy(query)
    
    def _execute_legacy(self, query: str) -> Dict[str, Any]:
        """Legacy execute method for backward compatibility."""
        if not EXCEL_CONTEXT:
            return {
                "response": "No Excel context data loaded. Please ensure context files are available.",
                "success": False,
                "agent": self.name
            }
        
        try:
            # Combine all context dataframes
            combined_dataframes = []
            for context_name, df in EXCEL_CONTEXT.items():
                df_copy = df.copy()
                df_copy['_context_source'] = context_name
                combined_dataframes.append(df_copy)
            
            # Combine dataframes
            if len(combined_dataframes) == 1:
                main_df = combined_dataframes[0]
            else:
                main_df = pd.concat(combined_dataframes, ignore_index=True, sort=False)
            
            # Build metadata text for the agent
            files_metadata_text = self._build_metadata_text(combined_dataframes)
            
            # Create pandas agent
            agent = create_pandas_dataframe_agent(
                self.llm,
                main_df,
                verbose=False,
                allow_dangerous_code=True,  # Required for pandas data analysis operations
                agent_type="openai-functions"
            )
            
            # Enhanced prompt with context
            enhanced_query = f"""{files_metadata_text}

USER QUESTION: {query}

Please provide a clear, conversational answer based on the audit data. 
If the data shows specific numbers, dates, or statuses, include them in your response.
Format your response as plain text (no markdown), and be specific about which audits, issues, or workpapers you're referring to."""
            
            # Execute query
            result = agent.invoke({"input": enhanced_query})
            response_text = result.get("output", str(result))
            
            # Clean up response
            if isinstance(response_text, str):
                # Remove any markdown formatting if present
                response_text = response_text.replace("```", "").strip()
            
            return {
                "response": response_text,
                "success": True,
                "agent": self.name,
                "context_used": "excel"
            }
            
        except Exception as e:
            return {
                "response": f"Error querying Excel context: {str(e)}",
                "success": False,
                "agent": self.name,
                "error": str(e)
            }
    
    def _execute_mastra(self, query: str, context: RuntimeContext) -> Dict[str, Any]:
        """Mastra-enabled execute method with runtime context."""
        if not EXCEL_CONTEXT:
            return {
                "response": "No Excel context data loaded. Please ensure context files are available.",
                "success": False,
                "agent": self.name
            }
        
        try:
            # Combine all context dataframes
            combined_dataframes = []
            for context_name, df in EXCEL_CONTEXT.items():
                df_copy = df.copy()
                df_copy['_context_source'] = context_name
                combined_dataframes.append(df_copy)
            
            # Combine dataframes
            if len(combined_dataframes) == 1:
                main_df = combined_dataframes[0]
            else:
                main_df = pd.concat(combined_dataframes, ignore_index=True, sort=False)
            
            # Build metadata text for the agent
            files_metadata_text = self._build_metadata_text(combined_dataframes)
            
            # Create pandas agent
            agent = create_pandas_dataframe_agent(
                self.llm,
                main_df,
                verbose=False,
                allow_dangerous_code=True,
                agent_type="openai-functions"
            )
            
            # Use Mastra's prompt building with runtime context
            enhanced_query = f"""{files_metadata_text}

{self._build_prompt(query, context)}

Please provide a clear, conversational answer based on the audit data. 
If the data shows specific numbers, dates, or statuses, include them in your response.
Format your response as plain text (no markdown), and be specific about which audits, issues, or workpapers you're referring to."""
            
            # Execute query
            result = agent.invoke({"input": enhanced_query})
            response_text = result.get("output", str(result))
            
            # Clean up response
            if isinstance(response_text, str):
                response_text = response_text.replace("```", "").strip()
            
            # Save to memory
            if context.thread_id:
                self.memory.add_message(context.thread_id, "assistant", response_text)
            
            return {
                "response": response_text,
                "success": True,
                "agent": self.name,
                "context_used": "excel",
                "resource_id": context.resource_id,
                "thread_id": context.thread_id
            }
            
        except Exception as e:
            return {
                "response": f"Error querying Excel context: {str(e)}",
                "success": False,
                "agent": self.name,
                "error": str(e)
            }
    
    def _execute(self, prompt: str, context: RuntimeContext, **kwargs) -> Dict[str, Any]:
        """MastraAgent interface implementation."""
        # Extract query from prompt (last line after "User Query:")
        query = prompt.split("User Query:")[-1].strip() if "User Query:" in prompt else prompt
        return self._execute_mastra(query, context)
    
    def _build_metadata_text(self, dataframes: List[pd.DataFrame]) -> str:
        """Build metadata text describing the Excel context."""
        text = f"\n{'='*80}\nAUDIT DATA CONTEXT\n{'='*80}\n"
        text += f"You are working with {len(dataframes)} data sources:\n\n"
        
        for i, df in enumerate(dataframes, 1):
            context_name = df['_context_source'].iloc[0] if '_context_source' in df.columns else f"source_{i}"
            text += f"\n{'─'*80}\n"
            text += f"SOURCE {i}: {context_name}\n"
            text += f"{'─'*80}\n"
            text += f"Total Rows: {len(df):,}\n"
            text += f"Columns ({len(df.columns)}): {', '.join([c for c in df.columns if c != '_context_source'])}\n"
        
        text += f"\n{'='*80}\n"
        text += "KEY RELATIONSHIPS:\n"
        text += "- All sources share 'Audit Title' - use this to join/relate data\n"
        text += "- Audits contain: Audit Title, Status, Dates, Managers, Ratings\n"
        text += "- Issues contain: Audit Title, Issue Title, Rating, Priority, Status\n"
        text += "- Workpapers contain: Audit Title, Workpaper Title, Status, Dates\n"
        
        return text


# ============================================================================
# DOCUMENT Q&A AGENT
# ============================================================================

class DocumentQAAgent(BaseAgent, MastraAgent):
    """Agent for answering questions about uploaded documents using RAG - Mastra-enabled."""
    
    def __init__(self):
        # Initialize BaseAgent for backward compatibility
        BaseAgent.__init__(
            self,
            name="document_qa",
            description="Answers questions about uploaded documents using RAG (Retrieval Augmented Generation)"
        )
        # Initialize MastraAgent with instructions and memory
        instructions = """You are an expert document analyst specializing in audit and GRC documents. Your role is to:
1. Answer questions about uploaded documents using RAG (Retrieval Augmented Generation)
2. Use audit terminology appropriately based on the audit system context
3. Provide accurate answers based on document content
4. If information is not in the documents, clearly state that
5. Reference specific sections or details when possible"""
        
        MastraAgent.__init__(
            self,
            name="document_qa",
            description="Answers questions about uploaded documents using RAG",
            instructions=instructions,
            memory=global_memory
        )
        self.llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    
    def can_handle(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Detect if query is about uploaded documents."""
        if not DOCUMENT_STORES:
            return False
        
        # Keywords that suggest document queries
        doc_keywords = [
            "document", "uploaded", "file", "summary", "what does the document",
            "what is in the document", "tell me about the document"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in doc_keywords)
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute document Q&A using RAG."""
        # Convert dict context to RuntimeContext if using Mastra
        mastra_context = None
        if context:
            mastra_context = RuntimeContext(
                resource_id=context.get("resource_id"),
                thread_id=context.get("thread_id"),
                metadata=context.get("metadata", {})
            )
        
        # Use Mastra generate method if context provided, otherwise use legacy method
        if mastra_context:
            return self._execute_mastra(query, mastra_context, context)
        else:
            return self._execute_legacy(query, context)
    
    def _execute_legacy(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Legacy execute method for backward compatibility."""
        if not DOCUMENT_STORES:
            return {
                "response": "No documents have been uploaded yet. Please upload documents first.",
                "success": False,
                "agent": self.name,
                "sources": []
            }
        
        # Get document IDs from context if provided
        document_ids = context.get("document_ids") if context else None
        
        # Determine which documents to query
        if document_ids:
            stores_to_query = {doc_id: DOCUMENT_STORES[doc_id] 
                              for doc_id in document_ids if doc_id in DOCUMENT_STORES}
        else:
            stores_to_query = DOCUMENT_STORES
        
        if not stores_to_query:
            return {
                "response": "No matching documents found.",
                "success": False,
                "agent": self.name,
                "sources": []
            }
        
        # Get Excel context for domain understanding
        audit_context = get_excel_context_summary()
        
        # Query each document store
        all_answers = []
        all_sources = []
        
        for doc_id, vector_store in stores_to_query.items():
            try:
                # Create retrieval QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                
                # Enhance query with audit context
                enhanced_query = f"""You are an expert audit and GRC (Governance, Risk, Compliance) assistant.

AUDIT SYSTEM CONTEXT (for understanding audit terminology):
{audit_context[:2000]}

USER QUESTION: {query}

Answer the question based on the document content. Use audit terminology appropriately.
If the answer is not in the documents, say so."""
                
                result = qa_chain.invoke({"query": enhanced_query})
                answer = result.get("result", "")
                source_docs = result.get("source_documents", [])
                
                if answer:
                    all_answers.append(answer)
                    all_sources.append(doc_id)
                    
            except Exception as e:
                print(f"Error querying document {doc_id}: {str(e)}")
                continue
        
        # Combine answers
        if not all_answers:
            return {
                "response": "No answer found in the documents.",
                "success": False,
                "agent": self.name,
                "sources": []
            }
        
        # Combine multiple answers if needed
        combined_answer = "\n\n".join(all_answers) if len(all_answers) > 1 else all_answers[0]
        
        return {
            "response": combined_answer,
            "success": True,
            "agent": self.name,
            "context_used": "document",
            "sources": all_sources
        }
    
    def _execute_mastra(self, query: str, context: RuntimeContext, legacy_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mastra-enabled execute method with runtime context."""
        if not DOCUMENT_STORES:
            return {
                "response": "No documents have been uploaded yet. Please upload documents first.",
                "success": False,
                "agent": self.name,
                "sources": []
            }
        
        # Get document IDs from context if provided
        document_ids = None
        if legacy_context:
            document_ids = legacy_context.get("document_ids")
        if not document_ids and context.metadata:
            document_ids = context.metadata.get("document_ids")
        
        # Determine which documents to query
        if document_ids:
            stores_to_query = {doc_id: DOCUMENT_STORES[doc_id] 
                              for doc_id in document_ids if doc_id in DOCUMENT_STORES}
        else:
            stores_to_query = DOCUMENT_STORES
        
        if not stores_to_query:
            return {
                "response": "No matching documents found.",
                "success": False,
                "agent": self.name,
                "sources": []
            }
        
        # Get Excel context for domain understanding
        audit_context = get_excel_context_summary()
        
        # Query each document store
        all_answers = []
        all_sources = []
        
        for doc_id, vector_store in stores_to_query.items():
            try:
                # Create retrieval QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                
                # Use Mastra's prompt building with runtime context
                mastra_prompt = self._build_prompt(query, context)
                
                # Enhance query with audit context
                enhanced_query = f"""You are an expert audit and GRC (Governance, Risk, Compliance) assistant.

AUDIT SYSTEM CONTEXT (for understanding audit terminology):
{audit_context[:2000]}

{mastra_prompt}

Answer the question based on the document content. Use audit terminology appropriately.
If the answer is not in the documents, say so."""
                
                result = qa_chain.invoke({"query": enhanced_query})
                answer = result.get("result", "")
                source_docs = result.get("source_documents", [])
                
                if answer:
                    all_answers.append(answer)
                    all_sources.append(doc_id)
                    
            except Exception as e:
                print(f"Error querying document {doc_id}: {str(e)}")
                continue
        
        # Combine answers
        if not all_answers:
            return {
                "response": "No answer found in the documents.",
                "success": False,
                "agent": self.name,
                "sources": []
            }
        
        # Combine multiple answers if needed
        combined_answer = "\n\n".join(all_answers) if len(all_answers) > 1 else all_answers[0]
        
        # Save to memory
        if context.thread_id:
            self.memory.add_message(context.thread_id, "assistant", combined_answer)
        
        return {
            "response": combined_answer,
            "success": True,
            "agent": self.name,
            "context_used": "document",
            "sources": all_sources,
            "resource_id": context.resource_id,
            "thread_id": context.thread_id
        }
    
    def _execute(self, prompt: str, context: RuntimeContext, **kwargs) -> Dict[str, Any]:
        """MastraAgent interface implementation."""
        # Extract query from prompt (last line after "User Query:")
        query = prompt.split("User Query:")[-1].strip() if "User Query:" in prompt else prompt
        return self._execute_mastra(query, context)


# ============================================================================
# AGENT ROUTER
# ============================================================================

class AgentRouter:
    """Routes queries to appropriate agents based on their capabilities."""
    
    def __init__(self):
        self.agents: List[BaseAgent] = []
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the router."""
        self.agents.append(agent)
        print(f"Registered agent: {agent.name} - {agent.description}")
    
    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route a query to the best matching agent(s).
        
        Args:
            query: User's query
            context: Optional context information
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.agents:
            return {
                "response": "No agents available.",
                "success": False
            }
        
        # Find agents that can handle the query
        capable_agents = [agent for agent in self.agents if agent.can_handle(query, context)]
        
        if not capable_agents:
            # Try all agents if none explicitly match
            capable_agents = self.agents
        
        # Try agents in order until one succeeds
        for agent in capable_agents:
            try:
                result = agent.execute(query, context)
                if result.get("success"):
                    return result
            except Exception as e:
                print(f"Agent {agent.name} failed: {str(e)}")
                continue
        
        # If no agent succeeded, try the first one as fallback
        if capable_agents:
            result = capable_agents[0].execute(query, context)
            return result
        
        return {
            "response": "Unable to process query. Please try rephrasing.",
            "success": False
        }
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents and their metadata."""
        return [agent.get_metadata() for agent in self.agents]


# ============================================================================
# MAIN AGENT SYSTEM
# ============================================================================

class AuditAssistantAgentSystem:
    """Main agent system for the audit assistant."""
    
    def __init__(self):
        self.router = AgentRouter()
        self._setup_agents()
    
    def _setup_agents(self):
        """Initialize and register all agents."""
        self.router.register_agent(ExcelQueryAgent())
        self.router.register_agent(DocumentQAAgent())
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query through the agent system.
        
        Args:
            query: User's query
            context: Optional context (e.g., document_ids, conversation_history)
            
        Returns:
            Dictionary with response and metadata
        """
        return self.router.route(query, context)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about available agents."""
        return {
            "agents": self.router.get_available_agents(),
            "total_agents": len(self.router.agents)
        }


# ============================================================================
# GLOBAL AGENT SYSTEM INSTANCE
# ============================================================================

# Agent system will be initialized after main.py loads
agent_system = None

def initialize_agent_system(excel_context, document_stores, openai_model, context_summary_func):
    """Initialize the agent system with shared state from main.py"""
    global EXCEL_CONTEXT, DOCUMENT_STORES, OPENAI_MODEL, get_excel_context_summary, agent_system
    
    EXCEL_CONTEXT = excel_context
    DOCUMENT_STORES = document_stores
    OPENAI_MODEL = openai_model
    get_excel_context_summary = context_summary_func
    
    agent_system = AuditAssistantAgentSystem()
    return agent_system
