"""
Mastra-inspired framework for AI agents.

This module provides a Mastra-like architecture with:
- Runtime context for dynamic agent behavior
- Memory/state management for conversation history
- Tools system for agent capabilities
- Workflow orchestration
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from collections import defaultdict


# ============================================================================
# RUNTIME CONTEXT
# ============================================================================

@dataclass
class RuntimeContext:
    """
    Runtime context for agents - allows dynamic configuration based on input.
    Similar to Mastra's runtime context feature.
    """
    resource_id: Optional[str] = None  # e.g., "audit-123"
    thread_id: Optional[str] = None  # e.g., "workpaper-session"
    metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context_summary(self) -> str:
        """Get a summary of the context for LLM prompts."""
        summary = []
        if self.resource_id:
            summary.append(f"Resource: {self.resource_id}")
        if self.thread_id:
            summary.append(f"Thread: {self.thread_id}")
        if self.metadata:
            summary.append(f"Metadata: {json.dumps(self.metadata, indent=2)}")
        if self.conversation_history:
            summary.append(f"Conversation history: {len(self.conversation_history)} messages")
        return "\n".join(summary) if summary else "No context"


# ============================================================================
# MEMORY/STATE MANAGEMENT
# ============================================================================

class MemoryStore:
    """
    Memory store for agent state and conversation history.
    Similar to Mastra's memory system.
    """
    
    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._threads: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def save(self, resource_id: str, thread_id: str, data: Dict[str, Any]):
        """Save state for a resource/thread combination."""
        key = f"{resource_id}:{thread_id}"
        self._storage[key] = data
        if "messages" in data:
            self._threads[thread_id] = data["messages"]
    
    def load(self, resource_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
        """Load state for a resource/thread combination."""
        key = f"{resource_id}:{thread_id}"
        return self._storage.get(key)
    
    def get_thread_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a thread."""
        return self._threads.get(thread_id, [])
    
    def add_message(self, thread_id: str, role: str, content: str):
        """Add a message to thread history."""
        self._threads[thread_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })


# ============================================================================
# TOOLS SYSTEM
# ============================================================================

@dataclass
class Tool:
    """A tool that an agent can use."""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return self.func(**kwargs)


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self._tools.values()
        ]


# ============================================================================
# MASTRA AGENT BASE
# ============================================================================

class MastraAgent:
    """
    Base class for Mastra-style agents.
    Supports runtime context, memory, and tools.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        memory: Optional[MemoryStore] = None,
        tools: Optional[List[Tool]] = None
    ):
        self.name = name
        self.description = description
        self.instructions = instructions
        self.memory = memory or MemoryStore()
        self.tool_registry = ToolRegistry()
        
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)
    
    def generate(
        self,
        query: str,
        context: Optional[RuntimeContext] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response to a query.
        Similar to Mastra's agent.generate() method.
        
        Args:
            query: User's query
            context: Runtime context (resource, thread, metadata)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with response and metadata
        """
        # Create default context if not provided
        if context is None:
            context = RuntimeContext()
        
        # Load conversation history from memory if thread_id exists
        if context.thread_id:
            history = self.memory.get_thread_history(context.thread_id)
            context.conversation_history = history
        
        # Add current query to context
        context.add_message("user", query)
        
        # Build enhanced prompt with context
        enhanced_prompt = self._build_prompt(query, context, **kwargs)
        
        # Execute agent logic (to be implemented by subclasses)
        response = self._execute(enhanced_prompt, context, **kwargs)
        
        # Save to memory
        if context.thread_id:
            self.memory.add_message(context.thread_id, "assistant", response.get("response", ""))
            self.memory.save(
                context.resource_id or "default",
                context.thread_id,
                {"messages": context.conversation_history}
            )
        
        return response
    
    def _build_prompt(self, query: str, context: RuntimeContext, **kwargs) -> str:
        """Build enhanced prompt with context and instructions."""
        prompt_parts = [
            f"Agent: {self.name}",
            f"Description: {self.description}",
            f"\nInstructions:\n{self.instructions}",
        ]
        
        if context.get_context_summary() != "No context":
            prompt_parts.append(f"\nContext:\n{context.get_context_summary()}")
        
        if context.conversation_history:
            prompt_parts.append("\nConversation History:")
            for msg in context.conversation_history[-5:]:  # Last 5 messages
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
        
        if self.tool_registry.list_tools():
            prompt_parts.append("\nAvailable Tools:")
            for tool_info in self.tool_registry.list_tools():
                prompt_parts.append(f"- {tool_info['name']}: {tool_info['description']}")
        
        prompt_parts.append(f"\n\nUser Query: {query}")
        prompt_parts.append("\nPlease provide a helpful response based on the context and instructions above.")
        
        return "\n".join(prompt_parts)
    
    def _execute(self, prompt: str, context: RuntimeContext, **kwargs) -> Dict[str, Any]:
        """
        Execute agent logic. To be implemented by subclasses.
        
        Returns:
            Dictionary with 'response' key and optional metadata
        """
        raise NotImplementedError("Subclasses must implement _execute method")
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent."""
        self.tool_registry.register(tool)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "tools": self.tool_registry.list_tools()
        }


# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================

class Workflow:
    """
    Workflow for orchestrating multiple agents and steps.
    Similar to Mastra's workflow system.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Callable] = []
    
    def add_step(self, step: Callable):
        """Add a step to the workflow."""
        self.steps.append(step)
    
    def execute(self, initial_input: Any, context: Optional[RuntimeContext] = None) -> Any:
        """Execute all steps in the workflow."""
        result = initial_input
        for step in self.steps:
            if context:
                result = step(result, context)
            else:
                result = step(result)
        return result


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

# Global memory store
global_memory = MemoryStore()

# Global tool registry
global_tool_registry = ToolRegistry()
