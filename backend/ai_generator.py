import anthropic
from typing import List, Optional, Dict, Any
from config import config

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to two tools for course information.

Available Tools:
1. **get_course_outline**: Use for queries about course structure, overview, or lesson list
   - Course outline or syllabus requests
   - "What lessons are in [course]?"
   - "What does [course] cover?"
   - Course instructor or link information

2. **search_course_content**: Use for queries about specific content within courses/lessons
   - Specific topics, concepts, or details from lessons
   - "What is covered in lesson X?"
   - Technical details or implementations
   - Searching for specific information across courses

Tool Usage Guidelines:
- **Up to 2 tool call rounds per query** - You can make sequential tool calls to gather information
- **Sequential reasoning**: Use first tool's results to inform second tool call if needed
- **Common patterns**:
  * Get course outline first, then search specific lesson content
  * Search content, then retrieve related course structure
  * Compare information across multiple courses/lessons
- **Efficient usage**: Only make additional tool calls when prior results are insufficient
- **Synthesize all tool results** into accurate, comprehensive responses
- **If any tool yields no results**, state this clearly without speculation

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course structure questions**: Use get_course_outline tool
- **Course content questions**: Use search_course_content tool
- **Complex queries**: Use multiple tools sequentially if needed
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, tool explanations, or query analysis
  - Do not mention "based on the tool results" or similar phrases
  - Do not describe your tool usage strategy


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle up to MAX_TOOL_ROUNDS of tool execution with state tracking.

        Rounds terminate when:
        - MAX_TOOL_ROUNDS completed
        - Claude's response has no tool_use blocks
        - Tool execution fails

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        MAX_TOOL_ROUNDS = config.MAX_TOOL_ROUNDS
        tool_round = 0

        # Initialize message history with user query
        messages = base_params["messages"].copy()
        current_response = initial_response

        # Loop through tool rounds
        while tool_round < MAX_TOOL_ROUNDS:
            tool_round += 1

            # Check if current response contains tool use
            if current_response.stop_reason != "tool_use":
                # Termination: No tool_use blocks
                return current_response.content[0].text

            # Add assistant's tool use response to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls and collect results
            tool_results = []

            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })

            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Prepare next API call parameters
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }

            # KEY: Continue passing tools if we haven't hit max rounds
            if tool_round < MAX_TOOL_ROUNDS:
                next_params["tools"] = base_params.get("tools", [])
                next_params["tool_choice"] = {"type": "auto"}

            # Make next API call
            current_response = self.client.messages.create(**next_params)

        # Termination: Max rounds reached
        return current_response.content[0].text