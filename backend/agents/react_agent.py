"""
Enhanced ReAct Agent using LangGraph for UAV Log Analysis
This module implements a true ReAct-based agent with planning and execution loops
using LangGraph for improved decision making and reasoning.
"""

import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Literal
import asyncio
import re

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnableConfig
)
from pydantic import BaseModel, Field

# LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

# Local imports
from telemetry.analyzer import TelemetryAnalyzer
from chat.memory_manager import EnhancedMemoryManager

# Type definitions for ReAct agent state
class AgentState(TypedDict):
    """State maintained in the agent throughout the conversation"""
    # Input fields
    query: str
    messages: List[Dict[str, Any]]
    telemetry_data: Dict[str, Any]
    session_id: str
    memory_context: Dict[str, Any]
    
    # Process fields
    thoughts: List[str]
    plan: List[str]
    observations: List[str]
    current_step: int
    
    # Output fields
    answer: Optional[str]
    analysis_results: Dict[str, Any]
    
    # Tool tracking
    tools_used: List[str]
    tools_results: List[Dict[str, Any]]
    
    # Error handling
    errors: List[str]
    
    # Status tracking
    status: Literal["THINKING", "PLANNING", "EXECUTING", "ANSWERING", "ERROR", "COMPLETE"]

# Tool schema definitions
class GetFlightMetricsSchema(BaseModel):
    """Schema for get_flight_metrics tool."""
    metric: str = Field(..., description="The metric to retrieve (altitude, speed, battery, etc.)")
    detailed: bool = Field(False, description="Whether to return detailed information or a summary")

class DetectAnomaliesSchema(BaseModel):
    """Schema for detect_anomalies tool."""
    parameter: str = Field(..., description="The parameter to check for anomalies (or 'all' for all parameters)")

class AnalyzeTimeRangeSchema(BaseModel):
    """Schema for analyze_time_range tool."""
    start_time: str = Field(..., description="The start time (timestamp or relative time)")
    end_time: str = Field(..., description="The end time (timestamp or relative time)")
    parameter: str = Field(..., description="The parameter to analyze")

class CorrelateParametersSchema(BaseModel):
    """Schema for correlate_parameters tool."""
    parameters: List[str] = Field(..., description="List of parameters to correlate")

class ExecuteDataQuerySchema(BaseModel):
    """Schema for execute_data_query tool."""
    query: str = Field(..., description="The query to execute (in natural language)")


class ReActAgent:
    """
    True ReAct-based agent for UAV log analysis with proper planning and execution loops.
    Uses LangGraph for improved decision making and integrates with analyzer for telemetry processing.
    """
    
    def __init__(
        self,
        session_id: str,
        telemetry_data: Dict,
        analyzer: TelemetryAnalyzer,
        memory_manager: Optional[EnhancedMemoryManager] = None,
        memory_window_size: int = 10
    ):
        self.session_id = session_id
        self.telemetry_data = telemetry_data
        self.analyzer = analyzer
        
        # Ensure OPENAI_API_KEY is set and use it directly
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set in ReActAgent!")
        
        # Initialize LLM with configurable options FIRST
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
            openai_api_key=openai_api_key,
            max_retries=2,
            request_timeout=30
        )

        # Initialize memory manager if not provided, passing the LLM
        if memory_manager is None:
            self.memory_manager = EnhancedMemoryManager(
                session_id=session_id,
                window_size=memory_window_size,
                llm=self.llm  # Pass the initialized LLM
            )
        else:
            self.memory_manager = memory_manager
        
        # System prompt - separate instructions from examples
        self.system_instructions = """
You are **MAVLink Analyst Pro**, an advanced agentic UAV-flight-data assistant.

# KEY RESPONSIBILITIES
1. Analyze telemetry data using a deliberate ReAct (Reasoning + Action) framework
2. Think step-by-step about complex UAV data analysis problems
3. Dynamically plan and execute flight data analysis based on user queries
4. Detect anomalies, patterns, and insights in UAV flight data

# AVAILABLE TOOLS
- get_flight_metrics: Retrieve specific flight metrics like altitude, speed, or battery
- detect_anomalies: Find anomalies in specific parameters or across all telemetry
- analyze_time_range: Analyze data within specific time periods
- correlate_parameters: Find correlations between multiple telemetry parameters
- execute_data_query: Run custom data analysis queries on telemetry data

# REACTION FRAMEWORK
For each query, carefully follow this strict ReAct framework:

1. THINK: Reason step-by-step through the query to understand what's being asked
2. PLAN: Create a clear, sequential plan of analysis steps to address the query
3. EXECUTE: Run the planned steps in sequence, updating based on what you learn
4. OBSERVE: Document your findings from each analysis step
5. CONCLUDE: Synthesize all findings into a comprehensive response

# FORMAT REQUIREMENTS
- Use JSON structure for all tool calls and outputs
- Always check results before proceeding to the next step
- Format numeric data consistently and with appropriate units
- Structure your final response with clear sections for different insights

# REASONING PRINCIPLES
- Prioritize safety-critical issues in your analysis
- Consider correlations between parameters (e.g., battery voltage drops during high-current maneuvers)
- Use contextual knowledge about UAV flight phases and normal operating parameters
- Think probabilistically about anomalies and their likely causes
- When uncertain, acknowledge limitations and suggest additional data that would help

# IMPORTANT REMINDER
Each time you use a tool, carefully observe the results BEFORE proceeding to next actions. Do not use tools unnecessarily, but do use them when needed to provide accurate data. Use ONLY real data from tool outputs in your answers, never fabricate values.
"""
        
        # Setup tools for the agent
        self.tools = self._create_tools()
        
        # Setup the ReAct agent graph
        self.workflow = self._create_agent_workflow()
        
        # Initialize checkpoint storage for workflow state
        self.memory_saver = MemorySaver()
        
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message through the ReAct agent workflow."""
        try:
            # Initialize state
            initial_state = self._initialize_state(message)
            
            # Run the workflow with timeout protection - shorter timeout to avoid hanging the server
            result = await asyncio.wait_for(
                self._async_run_workflow(initial_state),
                timeout=30  # 30 second timeout for full processing to stay under server timeout
            )
            
            # Store the conversation in memory
            await self.memory_manager.add_message(
                role="user",
                content=message,
                metadata={"query_type": "telemetry_analysis"}
            )
            
            if result.get("answer"):
                await self.memory_manager.add_message(
                    role="assistant",
                    content=result["answer"],
                    metadata={
                        "tools_used": result.get("tools_used", []),
                        "analysis_results": self._summarize_analysis(result.get("analysis_results", {}))
                    }
                )
            
            return {
                "response": result.get("answer", "I was unable to analyze the flight data."),
                "analysis": result.get("analysis_results", {}),
                "thought_process": result.get("thoughts", []),
                "tools_used": result.get("tools_used", [])
            }
            
        except asyncio.TimeoutError:
            # Create a fallback response with any partial results we have
            print("DEBUG: Workflow execution timed out, creating fallback answer")
            try:
                # Attempt to get partial results and form a simple answer
                partial_results = []
                
                if hasattr(self.memory_saver, "values") and self.session_id in self.memory_saver.values:
                    # Get last checkpoint
                    last_checkpoint = list(self.memory_saver.values[self.session_id].values())[-1]
                    if "tools_results" in last_checkpoint:
                        partial_results = last_checkpoint["tools_results"]
                
                # Create simple fallback answer with any available data
                fallback_answer = self._create_fallback_answer(message, partial_results)
                
                return {
                    "response": fallback_answer + "\n\nNote: Analysis was interrupted due to time constraints. Try a more specific query for better results.",
                    "analysis": {},
                    "thought_process": ["Analysis timed out - partial results only"],
                    "tools_used": []
                }
            except Exception as fallback_error:
                print(f"ERROR creating fallback response: {str(fallback_error)}")
                return {
                    "response": "Analysis timed out. Your query might be too complex for current processing limits. Please try a simpler or more specific query.",
                    "analysis": {},
                    "thought_process": ["Analysis timed out"],
                    "tools_used": []
                }
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Error in processing message: {str(e)}\n{traceback_str}")
            return {
                "response": "I encountered an error while analyzing the flight data. Please try a different or more specific question.",
                "analysis": {},
                "thought_process": [f"Error: {str(e)}"],
                "tools_used": []
            }
    
    def _initialize_state(self, message: str) -> AgentState:
        """Initialize the agent state with the user query."""
        return AgentState(
            query=message,
            messages=[{"role": "user", "content": message}],
            telemetry_data=self.telemetry_data,
            session_id=self.session_id,
            memory_context={},
            thoughts=[],
            plan=[],
            observations=[],
            current_step=0,
            answer=None,
            analysis_results={},
            tools_used=[],
            tools_results=[],
            errors=[],
            status="THINKING"
        )
    
    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create the tools for the agent to use."""
        return [
            {
                "name": "get_flight_metrics",
                "description": "Retrieve specific flight metrics from the telemetry data",
                "func": self._get_flight_metrics,
                "args": {
                    "metric": {"type": "string", "description": "The metric to retrieve (altitude, speed, battery, etc.)"},
                    "detailed": {"type": "boolean", "description": "Whether to return detailed information or a summary", "default": False}
                },
                "required": ["metric"]
            },
            {
                "name": "detect_anomalies",
                "description": "Detect anomalies in specific telemetry parameters",
                "func": self._detect_anomalies,
                "args": {
                    "parameter": {"type": "string", "description": "The parameter to check for anomalies (or 'all' for all parameters)"}
                },
                "required": ["parameter"]
            },
            {
                "name": "analyze_time_range",
                "description": "Analyze telemetry data within specific time range",
                "func": self._analyze_time_range,
                "args": {
                    "start_time": {"type": "string", "description": "The start time (timestamp or relative time)"},
                    "end_time": {"type": "string", "description": "The end time (timestamp or relative time)"},
                    "parameter": {"type": "string", "description": "The parameter to analyze"}
                },
                "required": ["start_time", "end_time", "parameter"]
            },
            {
                "name": "correlate_parameters",
                "description": "Find correlations between multiple telemetry parameters",
                "func": self._correlate_parameters,
                "args": {
                    "parameters": {"type": "array", "items": {"type": "string"}, "description": "List of parameters to correlate"}
                },
                "required": ["parameters"]
            },
            {
                "name": "execute_data_query",
                "description": "Run custom data analysis queries on telemetry data",
                "func": self._execute_data_query,
                "args": {
                    "query": {"type": "string", "description": "The query to execute (in natural language)"}
                },
                "required": ["query"]
            }
        ]
    
    def _create_agent_workflow(self) -> StateGraph:
        """Create the agent workflow graph using LangGraph."""
        # Setup the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each state
        workflow.add_node("thinking", self._thinking_step)
        workflow.add_node("planning", self._planning_step)
        workflow.add_node("executing", self._executing_step)
        workflow.add_node("observing", self._observing_step)
        workflow.add_node("answering", self._answering_step)
        
        # Add edges to connect the nodes
        workflow.add_edge("thinking", "planning")
        workflow.add_edge("planning", "executing")
        workflow.add_edge("executing", "observing")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "observing",
            self._should_continue_executing,
            {
                "continue": "executing",
                "complete": "answering"
            }
        )
        
        workflow.add_edge("answering", END)
        
        # Set the entry point
        workflow.set_entry_point("thinking")
        
        return workflow
    
    def _get_system_prompt_with_examples(self) -> str:
        """Return the complete system prompt with examples without variable extraction issues."""
        examples = """
# REQUIRED EXAMPLES - USE THESE PATTERNS EXACTLY

## Example 1: Maximum Altitude
User: What was the maximum altitude reached during this flight?

THINK: The user wants to know the maximum altitude reached during the flight. I need to retrieve altitude metrics from the telemetry data.

PLAN:
1. Use get_flight_metrics tool to retrieve altitude data
2. Extract the maximum altitude value from the metrics
3. Provide a comprehensive answer with context

EXECUTE: I'll use the get_flight_metrics tool to get altitude data.

TOOL: get_flight_metrics
ARGS: {"metric": "altitude", "detailed": true}

OBSERVATION: 
```
{
  "altitude": {
    "max": 63.3,
    "min": -2.7,
    "mean": 32.1,
    "std": 12.5,
    "field_used": "GLOBAL_POSITION_INT_relative_alt"
  }
}
```

ANSWER: The maximum altitude reached during this flight was 63.3 meters. The drone also had a minimum altitude of -2.7 meters (likely during takeoff calibration) and an average altitude of 32.1 meters throughout the flight.

## Example 2: GPS Signal Issues
User: Can you spot any issues in the GPS data?

THINK: The user wants me to identify potential issues or anomalies in the GPS data. I should check for GPS signal losses, accuracy issues, or other anomalies in GPS-related parameters.

PLAN:
1. Use the detect_anomalies tool to check for GPS-specific anomalies
2. Analyze GPS-related metrics using get_flight_metrics
3. Correlate GPS data with other parameters if needed for context

EXECUTE: First, I'll check for GPS anomalies.

TOOL: detect_anomalies
ARGS: {"parameter": "gps"}

OBSERVATION:
```
Detected 2 anomalies for 'gps':
1. Type: gps_signal_loss, Severity: 2.4, Timestamp: 857.42
   Metrics: {"duration": 1.8, "altitude": 78.3, "velocity": 15.6}
2. Type: gps_accuracy_degradation, Severity: 1.9, Timestamp: 1730.18
   Metrics: {"duration": 1.2, "altitude": 92.1, "velocity": 12.3}
```

EXECUTE: Now let's get more details on GPS metrics.

TOOL: get_flight_metrics
ARGS: {"metric": "gps", "detailed": true}

OBSERVATION:
```
{
  "GPS_RAW_INT_satellites_visible": {
    "max": 12,
    "min": 5,
    "mean": 9.2,
    "std": 1.7
  },
  "GPS_RAW_INT_eph": {
    "max": 240,
    "min": 120,
    "mean": 145.3,
    "std": 28.7
  }
}
```

EXECUTE: Let's analyze the time period around the first anomaly.

TOOL: analyze_time_range
ARGS: {"start_time": "857", "end_time": "859", "parameter": "GPS_RAW_INT_eph"}

OBSERVATION:
```
GPS_RAW_INT_eph analysis from 857.0 to 859.0:
- Maximum: 240 at timestamp 857.8
- Minimum: 120 at timestamp 858.5
- Average: 178.3
```

ANSWER: I've identified two significant issues in the GPS data:

1. GPS signal loss at timestamp 857.42:
   - Duration: 1.8 seconds
   - Severity: 2.4/3.0 (high)
   - Occurred at 78.3m altitude while moving at 15.6 m/s
   - HDOP (horizontal dilution of precision) spiked to 240 during this period (values over 200 indicate poor accuracy)

2. GPS accuracy degradation at timestamp 1730.18:
   - Duration: 1.2 seconds
   - Severity: 1.9/3.0 (moderate)
   - Occurred at 92.1m altitude while moving at 12.3 m/s

The satellite count ranged from 5-12 visible satellites throughout the flight, with an average of 9.2. The drop to 5 satellites likely contributed to the accuracy issues. The timing of the GPS issues suggests possible interference during high-speed maneuvers.

## Example 3: Anomaly Detection
User: Are there any anomalies in this flight?

THINK: The user wants a comprehensive analysis of all anomalies or irregular behavior detected in the flight data. I should look for anomalies across all telemetry parameters.

PLAN:
1. Use detect_anomalies tool to scan all parameters for anomalies
2. Get flight metrics to provide context for any anomalies found
3. Analyze specific time periods where anomalies occurred if needed

EXECUTE: Let's check for anomalies across all parameters.

TOOL: detect_anomalies
ARGS: {"parameter": "all"}

OBSERVATION:
```
Detected 5 anomalies:
1. Type: gps_signal_loss, Severity: 2.4, Timestamp: 612.3
   Metrics: {"duration": 3.2, "altitude": 82.1}
2. Type: rc_signal_loss, Severity: 2.8, Timestamp: 895.7
   Metrics: {"duration": 0.8, "altitude": 95.2}
3. Type: ekf_error, Severity: 2.6, Timestamp: 1421.3
   Metrics: {"error_type": "EKF_YAW_ERR", "altitude": 105.8}
4. Type: battery_anomaly, Severity: 2.2, Timestamp: 2345.6
   Metrics: {"voltage_drop": 0.8, "altitude": 62.4}
5. Type: gps_anomaly, Severity: 1.9, Timestamp: 2687.2
   Metrics: {"duration": 1.1, "altitude": 43.2}
```

EXECUTE: Let's get flight metrics for additional context.

TOOL: get_flight_metrics
ARGS: {"metric": "overview", "detailed": true}

OBSERVATION:
```
{
  "flight_time": "11 min 32 sec",
  "max_altitude": 105.8,
  "max_speed": 18.3,
  "battery_end": 76.2,
  "distance_traveled": 3215.8
}
```

ANSWER: I've identified 5 significant anomalies during this 11-minute flight:

1. RC Signal Loss (Most Severe - 2.8/3.0):
   - Occurred at timestamp 895.7 (about 7.5 minutes into flight)
   - Duration: 0.8 seconds
   - Altitude: 95.2 meters
   - This represents a momentary loss of direct control from the remote controller

2. EKF Error (Severity 2.6/3.0):
   - Occurred at timestamp 1421.3
   - Type: EKF_YAW_ERR (Extended Kalman Filter yaw error)
   - Altitude: 105.8 meters (at maximum flight altitude)
   - This indicates an issue with the drone's navigation system's yaw estimation

3. GPS Signal Loss (Severity 2.4/3.0):
   - Occurred at timestamp 612.3
   - Duration: 3.2 seconds
   - Altitude: 82.1 meters

4. Battery Anomaly (Severity 2.2/3.0):
   - Occurred at timestamp 2345.6
   - Sudden voltage drop of 0.8V
   - Altitude: 62.4 meters

5. GPS Accuracy Degradation (Severity 1.9/3.0):
   - Occurred at timestamp 2687.2
   - Duration: 1.1 seconds
   - Altitude: 43.2 meters

The RC signal loss is particularly concerning as it indicates a temporary loss of direct control. The EKF yaw error occurring at maximum altitude could indicate magnetic interference at that height. These anomalies suggest reviewing the drone's hardware and possibly updating firmware.

## Example 4: RC Signal Loss
User: When was the first instance of RC signal loss?

THINK: The user wants to know when the first RC (Remote Control) signal loss occurred. I need to look for RC signal anomalies and identify the earliest occurrence.

PLAN:
1. Use detect_anomalies to find RC signal-related anomalies
2. Order the anomalies by timestamp to find the earliest one
3. Provide context for when this occurred during the flight

EXECUTE: Let's check for RC signal-related anomalies.

TOOL: detect_anomalies
ARGS: {"parameter": "rc"}

OBSERVATION:
```
Detected 2 anomalies for 'rc':
1. Type: rc_signal_loss, Severity: 2.8, Timestamp: 895.7
   Metrics: {"duration": 0.8, "altitude": 95.2, "velocity": 16.3}
2. Type: rc_signal_degradation, Severity: 1.7, Timestamp: 1423.4
   Metrics: {"duration": 1.2, "altitude": 103.1, "velocity": 14.8}
```

ANSWER: The first instance of RC signal loss occurred at timestamp 895.7, approximately 7.5 minutes into the flight. This was a complete signal loss with a severity rating of 2.8/3.0, lasting 0.8 seconds. At the time, the drone was flying at an altitude of 95.2 meters with a velocity of 16.3 m/s. A second, less severe RC signal degradation (not complete loss) occurred later at timestamp 1423.4.

## Example 5: Flight Time
User: How long was the total flight time?

THINK: The user wants to know the total duration of the flight from takeoff to landing. I need to retrieve flight metrics that include timing information.

PLAN:
1. Use get_flight_metrics to retrieve flight time data

EXECUTE: I'll use get_flight_metrics to get the flight time.

TOOL: get_flight_metrics
ARGS: {"metric": "flight_time"}

OBSERVATION:
```
{
  "flight_time": "11 min 32 sec",
  "start_timestamp": 1234.56,
  "end_timestamp": 1926.84
}
```

ANSWER: The total flight time was 11 minutes and 32 seconds, from timestamp 1234.56 to 1926.84.

## Example 6: Battery Temperature
User: What was the maximum battery temperature?

THINK: The user wants to know the maximum temperature that the battery reached during the flight. I need to retrieve battery-related metrics.

PLAN:
1. Use get_flight_metrics to retrieve battery data including temperature information

EXECUTE: I'll get the battery metrics.

TOOL: get_flight_metrics
ARGS: {"metric": "battery", "detailed": true}

OBSERVATION:
```
{
  "SYS_STATUS_voltage_battery": {
    "max": 12.6,
    "min": 11.8,
    "mean": 12.1,
    "std": 0.2
  },
  "SYS_STATUS_current_battery": {
    "max": 24.3,
    "min": 4.1,
    "mean": 14.8,
    "std": 5.3
  },
  "SYS_STATUS_battery_temperature": {
    "max": 43.2,
    "min": 28.5,
    "mean": 36.8,
    "std": 3.7
  }
}
```

ANSWER: The maximum battery temperature during the flight was 43.2°C. The temperature ranged from 28.5°C to 43.2°C, with an average of 36.8°C.

## Example 7: General Flight Overview
User: Tell me about this flight

THINK: The user wants a comprehensive overview of the flight. I should gather general metrics, key events, and anomalies to provide a complete picture.

PLAN:
1. Use get_flight_metrics to retrieve general flight metrics
2. Check for anomalies using detect_anomalies
3. Analyze specific interesting parameters as needed

EXECUTE: Let's start with general flight metrics.

TOOL: get_flight_metrics
ARGS: {"metric": "overview"}

OBSERVATION:
```
{
  "flight_time": "11 min 32 sec",
  "max_altitude": 105.8,
  "max_speed": 18.3,
  "battery_start": 98.5,
  "battery_end": 76.2,
  "distance_traveled": 3215.8,
  "avg_groundspeed": 12.4
}
```

EXECUTE: Now let's check for any anomalies during the flight.

TOOL: detect_anomalies
ARGS: {"parameter": "all"}

OBSERVATION:
```
Detected 3 anomalies:
1. Type: gps_signal_loss, Severity: 2.4, Timestamp: 612.3
   Metrics: {"duration": 3.2, "altitude": 82.1}
2. Type: rc_signal_loss, Severity: 2.8, Timestamp: 895.7
   Metrics: {"duration": 0.8, "altitude": 95.2}
3. Type: battery_anomaly, Severity: 2.2, Timestamp: 2345.6
   Metrics: {"voltage_drop": 0.8, "altitude": 62.4}
```

ANSWER: This flight lasted 11 minutes and 32 seconds, covering a distance of 3.2 kilometers at an average speed of 12.4 m/s. The drone reached a maximum altitude of 105.8 meters and a top speed of 18.3 m/s.

The battery level decreased from 98.5% to 76.2% during the flight, with one notable voltage drop at timestamp 2345.6.

The flight experienced 3 anomalies:
1. A brief RC signal loss (0.8 seconds) at timestamp 895.7 while at 95.2 meters altitude
2. A GPS signal loss lasting 3.2 seconds at timestamp 612.3
3. A battery anomaly with a voltage drop of 0.8V at timestamp 2345.6

Overall, despite these minor issues, the flight appears to have completed successfully with adequate battery reserves remaining.
"""
        
        # Combine the instructions with examples
        return self.system_instructions + examples
        
    def _thinking_step(self, state: AgentState) -> AgentState:
        """Think about the query and understand what's being asked."""
        # Create messages directly without using templates
        messages = [
            SystemMessage(content=self._get_system_prompt_with_examples()),
            HumanMessage(content=f"Please analyze this query about UAV flight data: {state['query']}"),
            SystemMessage(content="First, think carefully about what is being asked and what telemetry data you would need to answer this query effectively. Break down the query into key components.")
        ]
        
        # Call LLM directly with messages
        thinking_response = self.llm.invoke(messages)
        thoughts = self._parse_thinking(thinking_response.content)
        
        # Update state
        state["thoughts"].append(thoughts)
        state["status"] = "PLANNING"
        
        return state
    
    def _planning_step(self, state: AgentState) -> AgentState:
        """Create a plan to answer the query."""
        messages = [
            SystemMessage(content=self._get_system_prompt_with_examples()),
            HumanMessage(content=f"Please analyze this query about UAV flight data: {state['query']}"),
            AIMessage(content=f"My thought process: {state['thoughts'][-1] if state['thoughts'] else ''}"),
            SystemMessage(content="Now, create a step-by-step plan to analyze the flight data and answer this query. What specific tools would you use in what sequence?")
        ]
        
        # Call LLM directly with messages
        planning_response = self.llm.invoke(messages)
        plan = self._parse_plan(planning_response.content)
        
        # Update state
        state["plan"] = plan
        state["current_step"] = 0
        state["status"] = "EXECUTING"
        
        return state
    
    def _executing_step(self, state: AgentState) -> AgentState:
        """Execute the current step in the plan."""
        # Get current step
        if state["current_step"] >= len(state["plan"]):
            print("DEBUG: No more steps to execute, moving to answering")
            state["status"] = "ANSWERING"
            return state
            
        current_step = state["plan"][state["current_step"]]
        print(f"DEBUG: Executing step {state['current_step'] + 1}: {current_step}")
        
        # Add context from previous observations to help the model make better tool choices
        context_info = ""
        if state["observations"]:
            # Add the last observation as context to help with parameter selection
            last_observation = state["observations"][-1]
            if len(last_observation) > 500:
                context_info = f"Previous observation (truncated): {last_observation[:500]}...\n\n"
            else:
                context_info = f"Previous observation: {last_observation}\n\n"
        
        # Create messages directly instead of using ChatPromptTemplate
        plan_summary = "\n".join(f"{i+1}. {step}" for i, step in enumerate(state["plan"]))
        
        messages = [
            SystemMessage(content="""You are a UAV flight analyst who chooses the right tool to execute based on a plan step.
Your response should ALWAYS follow this format EXACTLY:

EXECUTE: [Brief explanation of what you're doing]

TOOL: [tool_name]
ARGS: [JSON-formatted arguments for the tool]

Do not include any other text. Choose from these tools ONLY:
- get_flight_metrics: Get metrics like altitude, speed, battery (args: metric, detailed)
- detect_anomalies: Find anomalies (args: parameter)
- analyze_time_range: Analyze data in time window (args: start_time, end_time, parameter)
- correlate_parameters: Find correlations (args: parameters[])
- execute_data_query: Run custom queries (args: query)

IMPORTANT TIPS:
1. When using analyze_time_range, ALWAYS use specific numeric timestamps, not placeholder text
2. For correlate_parameters, use specific field names, not general terms
3. For anomaly detection, try specific parameters like "altitude", "gps", "battery"
4. For get_flight_metrics, use "overview" to get general metrics or specific parameters

Examples:

EXECUTE: I'll get altitude data.

TOOL: get_flight_metrics
ARGS: {"metric": "altitude", "detailed": true}

OR

EXECUTE: I'll check for GPS anomalies.

TOOL: detect_anomalies
ARGS: {"parameter": "gps"}"""),
            HumanMessage(content=f"""Query: {state["query"]}
Plan: {plan_summary}
Current step: {current_step}
{context_info}            
Choose the appropriate tool to execute this step.""")
        ]
        
        # Execute directly without template variable extraction
        try:
            response = self.llm.invoke(messages)
            
            print(f"DEBUG: Tool selection response: {response.content}")
            
            # Parse the response to extract tool name and arguments
            tool_info = self._extract_tool_info(response.content)
            
            # Check if we got a valid tool
            if tool_info and "tool" in tool_info and "args" in tool_info:
                tool_name = tool_info["tool"]
                tool_args = tool_info["args"]
                
                try:
                    # Find the matching tool by name
                    matching_tool = None
                    for tool in self.tools:
                        if tool.get("name").lower() == tool_name.lower():
                            matching_tool = tool
                            break
                    
                    if matching_tool:
                        print(f"DEBUG: Executing tool {tool_name} with args {tool_args}")
                        # Execute the tool directly
                        result = matching_tool.get("func")(**tool_args)
                        print(f"DEBUG: Tool result: {result[:200]}..." if isinstance(result, str) and len(result) > 200 else f"DEBUG: Tool result: {result}")
                    else:
                        # No matching tool found, report error
                        error_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join([t.get('name') for t in self.tools])}"
                        state["errors"].append(error_msg)
                        result = f"Error: {error_msg}"
                    
                    # Update state
                    state["tools_used"].append(tool_name)
                    state["tools_results"].append({
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": result
                    })
                    
                except Exception as e:
                    # Handle tool execution errors
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    print(f"DEBUG: {error_msg}")
                    state["errors"].append(error_msg)
                    result = f"Error: {str(e)}"
                    
                    # Add the failed tool anyway so we have a record and can continue
                    state["tools_used"].append(tool_name)
                    state["tools_results"].append({
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": result
                    })
            else:
                print("DEBUG: No valid tool call found in response")
                result = "No tool was called. Please specify a valid tool and arguments."
                
                # Add a stub tool result so we don't get stuck
                state["tools_used"].append("error")
                state["tools_results"].append({
                    "tool": "error",
                    "arguments": {},
                    "result": result
                })
            
            # Store the result for observation
            state["observations"].append(result)
            state["status"] = "OBSERVING"
            
        except Exception as e:
            # Handle any other errors in executing step
            error_msg = f"Error in executing step: {str(e)}"
            print(f"DEBUG: {error_msg}")
            state["errors"].append(error_msg)
            state["observations"].append(f"Error: {str(e)}")
            state["status"] = "OBSERVING"  # Continue to observing step even with errors
            
            # Add an error tool result
            state["tools_used"].append("error")
            state["tools_results"].append({
                "tool": "error",
                "arguments": {},
                "result": f"Error: {str(e)}"
            })
        
        return state
    
    def _extract_tool_info(self, response_text: str) -> Dict[str, Any]:
        """Extract tool information from the response text."""
        try:
            if not response_text:
                return {}
                
            # Try to extract TOOL: and ARGS: sections
            tool_match = re.search(r'TOOL:\s*(\w+)', response_text)
            if not tool_match:
                # Try alternate format
                tool_match = re.search(r'tool\s*:\s*(\w+)', response_text, re.IGNORECASE)
                if not tool_match:
                    return {}
            
            tool_name = tool_match.group(1).strip()
            
            # Look for JSON args
            args_str = ""
            args_match = re.search(r'ARGS:\s*({.+})', response_text, re.DOTALL)
            if args_match:
                args_str = args_match.group(1).strip()
            else:
                # Try alternate format
                args_match = re.search(r'args\s*:\s*({.+})', response_text, re.DOTALL | re.IGNORECASE)
                if args_match:
                    args_str = args_match.group(1).strip()
                else:
                    return {}
            
            # Safely parse the args
            try:
                args = json.loads(args_str)
                return {
                    "tool": tool_name,
                    "args": args
                }
            except json.JSONDecodeError:
                print(f"DEBUG: Failed to parse args JSON: {args_str}")
                return {}
            
        except Exception as e:
            print(f"DEBUG: Error extracting tool info: {str(e)}")
            return {}
    
    def _observing_step(self, state: AgentState) -> AgentState:
        """Process the observations from tool execution."""
        # Get current observations
        current_observation = state["observations"][-1] if state["observations"] else "No observations yet."
        
        # Create messages directly instead of using ChatPromptTemplate
        plan_summary = "\n".join(f"{i+1}. {step}" for i, step in enumerate(state["plan"]))
        current_step_text = state["plan"][state["current_step"]] if state["current_step"] < len(state["plan"]) else "Final step"
        
        messages = [
            SystemMessage(content=self._get_system_prompt_with_examples()),
            HumanMessage(content=f"Query: {state['query']}"),
            AIMessage(content=f"Plan: {plan_summary}\nCurrent step: {current_step_text}"),
            SystemMessage(content=f"Analyze this observation from the current step: {current_observation}")
        ]
        
        # Process the observation
        observation_analysis = self.llm.invoke(messages)
        
        # Update current step
        state["current_step"] += 1
        
        # Update analysis results with observation
        current_tool = state["tools_used"][-1] if state["tools_used"] else "unknown"
        if current_tool in state["analysis_results"]:
            state["analysis_results"][current_tool].append(current_observation)
        else:
            state["analysis_results"][current_tool] = [current_observation]
        
        return state
    
    def _should_continue_executing(self, state: AgentState) -> str:
        """Determine if we should continue executing or move to answering."""
        # Too many errors, move to answering
        if state["errors"] and len(state["errors"]) > 3:
            print("DEBUG: Ending execution due to too many errors")
            return "complete"
        
        # Completed all steps in the plan
        if state["current_step"] >= len(state["plan"]):
            print("DEBUG: Ending execution due to completing all steps in the plan")
            return "complete"
        
        # If we've executed 3 or more steps, let's move to answering to avoid timeouts
        if len(state["tools_used"]) >= 3:
            print("DEBUG: Ending execution after 3 tool executions to avoid timeouts")
            return "complete"
            
        # Continue executing
        return "continue"
    
    def _answering_step(self, state: AgentState) -> AgentState:
        """Create the final answer based on all observations."""
        # Cap the number of results to avoid timeouts
        max_results = 3
        
        # Check if we have any useful tool results
        has_useful_data = False
        for result in state.get("tools_results", [])[:max_results]:
            if result and "result" in result and result["result"]:
                has_useful_data = True
                break
        
        # If no tools were used or no useful data, explicitly fetch key metrics now
        if not has_useful_data:
            print("DEBUG: No useful tool data found, explicitly gathering metrics")
            try:
                # Get altitude data directly from analyzer
                altitude_data = self.analyzer._analyze_altitude()
                if altitude_data and "statistics" in altitude_data:
                    max_alt = altitude_data["statistics"].get("max")
                    min_alt = altitude_data["statistics"].get("min")
                    
                    # Add it to our tools_results
                    state["tools_results"].append({
                        "tool": "get_flight_metrics",
                        "arguments": {"metric": "altitude"},
                        "result": f"Maximum altitude: {max_alt} meters, Minimum altitude: {min_alt} meters"
                    })
                    
                    # Make sure we record that we used the tool
                    if "get_flight_metrics" not in state["tools_used"]:
                        state["tools_used"].append("get_flight_metrics")
                    
                    print(f"DEBUG: Added altitude metrics: max={max_alt}m, min={min_alt}m")
                
                # Get other important metrics (limit to 1 to avoid timeout)
                try:
                    overview_data = self.analyzer.analyze_for_query("get overview")
                    if overview_data:
                        state["tools_results"].append({
                            "tool": "get_flight_metrics",
                            "arguments": {"metric": "overview"},
                            "result": json.dumps(overview_data, indent=2, default=str)
                        })
                except Exception as e:
                    print(f"Error fetching overview: {str(e)}")
                
            except Exception as e:
                print(f"ERROR gathering metrics: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # Gather all tool results (limit to avoid timeouts)
        tools_summary = []
        for i, result in enumerate(state["tools_results"][:max_results]):
            tool_name = result.get("tool", "unknown")
            tool_result = result.get("result", "No result")
            tool_args = result.get("arguments", {})
            
            # Trim very long results
            if isinstance(tool_result, str) and len(tool_result) > 1000:
                tool_result = tool_result[:1000] + "... (truncated)"
            
            # Summarize the tool call
            summary = f"Tool {i+1}: {tool_name}\n"
            summary += f"Arguments: {json.dumps(tool_args)}\n"
            summary += f"Result: {tool_result if isinstance(tool_result, str) else json.dumps(tool_result, default=str)}"
            
            tools_summary.append(summary)
        
        # Add a summary of how many tool results were capped
        if len(state["tools_results"]) > max_results:
            extra_results = len(state["tools_results"]) - max_results
            tools_summary.append(f"Note: {extra_results} additional tool results were omitted to prevent timeout.")
        
        # Create answer messages with conciseness instructions to avoid large responses
        messages = [
            SystemMessage(content="""
You are an experienced UAV flight analyst. Answer the user's query based ONLY on the provided flight data.
IMPORTANT: 
1. Use ONLY the actual data provided in the analysis results. Do NOT make up or hallucinate any values.
2. If the maximum altitude is shown as 63.3 meters, report that exact value, not 1200 meters or any other fabricated number.
3. BE EXTREMELY CONCISE. Users need quick, direct answers to avoid timeouts.
4. List only the key facts directly relevant to the user's question.
5. Avoid long explanations, introductions, or technical jargon.

Format your answer as:
- Direct factual answer to the query
- 1-2 relevant insights if applicable
- Any critical safety issues or anomalies detected
"""),
            HumanMessage(content=f"Query: {state['query']}"),
            SystemMessage(content=f"Analysis results:\n{chr(10).join(tools_summary)}")
        ]
        
        try:
            # Generate the answer with a specific timeout
            answer = self.llm.invoke(messages)
            
            # Update state
            state["answer"] = answer.content
            state["status"] = "COMPLETE"
        except Exception as e:
            # Create a fallback answer based on what we have
            error_msg = f"Error generating final answer: {str(e)}"
            print(f"DEBUG: {error_msg}")
            
            # Fallback answer using existing data
            fallback_answer = self._create_fallback_answer(state["query"], state["tools_results"][:max_results])
            state["answer"] = fallback_answer
            state["status"] = "COMPLETE"
        
        return state
    
    def _create_fallback_answer(self, query: str, tool_results: List[Dict[str, Any]]) -> str:
        """Create a simple fallback answer when the full answer generation fails."""
        try:
            # Extract key facts from tool results
            facts = []
            
            for result in tool_results:
                tool = result.get("tool", "")
                result_text = result.get("result", "")
                
                if tool == "get_flight_metrics" and isinstance(result_text, str):
                    # Try to extract altitude, speed, etc.
                    for metric in ["altitude", "max", "min", "speed", "battery"]:
                        if metric in result_text:
                            # Find the line containing the metric
                            for line in result_text.split("\n"):
                                if metric in line.lower():
                                    facts.append(line.strip())
                                    break
                
                elif tool == "detect_anomalies" and isinstance(result_text, str):
                    # Extract anomaly count
                    if "Detected" in result_text and "anomalies" in result_text:
                        first_line = result_text.split("\n")[0]
                        facts.append(first_line)
            
            # Create a simple response
            if facts:
                response = f"Based on the flight data analysis:\n- " + "\n- ".join(facts[:5])
            else:
                response = "Based on limited data, we cannot provide a complete answer to your query."
            
            return response
            
        except Exception as e:
            print(f"Error creating fallback answer: {str(e)}")
            return "Unable to analyze the flight data due to technical issues. Please try a simpler query."
    
    # Tool implementation methods
    def _get_flight_metrics(self, metric: str, detailed: bool = False) -> str:
        """Get specific flight metrics from telemetry data."""
        try:
            metrics = self.analyzer.analyze_for_query(f"get {metric} metrics")
            
            # Convert timestamps and any other non-serializable objects to strings
            def json_serializable(obj):
                if hasattr(obj, 'isoformat'):  # Handle datetime objects
                    return obj.isoformat()
                elif hasattr(obj, 'timestamp'):  # Handle pandas Timestamp
                    return obj.timestamp()
                else:
                    return str(obj)  # Convert any other non-serializable object to string
            
            if detailed:
                return json.dumps(metrics, indent=2, default=json_serializable)
            else:
                # Return a simplified summary
                if isinstance(metrics, dict):
                    summary = {}
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            summary[key] = {k: v for k, v in value.items() if k in ['max', 'min', 'mean', 'count']}
                        else:
                            summary[key] = value
                    return json.dumps(summary, indent=2, default=json_serializable)
                return json.dumps(metrics, indent=2, default=json_serializable)
        except Exception as e:
            error_msg = f"Error getting flight metrics for {metric}: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg
    
    def _detect_anomalies(self, parameter: str) -> str:
        """Detect anomalies in the specified parameter."""
        try:
            if parameter.lower() == "all":
                # Comprehensive anomaly detection
                result = self.analyzer.analyze_for_query("detect all anomalies")
            else:
                # Parameter-specific anomaly detection
                result = self.analyzer.analyze_for_query(f"analyze {parameter} anomalies")
            
            if "anomalies" in result and result["anomalies"]:
                anomalies = result["anomalies"]
                formatted_anomalies = []
                
                for i, anomaly in enumerate(anomalies, 1):
                    # Extract timestamp for use in other tools
                    ts = anomaly.get('timestamp')
                    timestamp_val = ts.timestamp() if hasattr(ts, 'timestamp') else str(ts)
                    
                    formatted = f"{i}. Type: {anomaly.get('type', 'unknown')}, "
                    formatted += f"Severity: {anomaly.get('severity', 'unknown')}, "
                    formatted += f"Timestamp: {timestamp_val}\n"
                    formatted += f"   Metrics: {json.dumps(anomaly.get('data', {}), default=str)}"
                    formatted_anomalies.append(formatted)
                
                return f"Detected {len(anomalies)} anomalies for '{parameter}':\n" + "\n".join(formatted_anomalies[:5]) + f"\n(showing first 5 of {len(anomalies)} anomalies)"
            else:
                return f"No anomalies detected for '{parameter}'."
        except Exception as e:
            error_msg = f"Error detecting anomalies for {parameter}: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg
    
    def _analyze_time_range(self, start_time: str, end_time: str, parameter: str) -> str:
        """Analyze telemetry data within a specific time range."""
        # Handle literal strings that might be passed as placeholders
        if start_time in ["anomaly_start_time", "start_time"] or end_time in ["anomaly_end_time", "end_time"]:
            # Get a reasonable default time range instead
            timestamps = self.analyzer.time_series.get("timestamp", [])
            if not timestamps:
                return "No timestamp data available for analysis."
            
            # Use first and last timestamps or a range in the middle
            if len(timestamps) > 1:
                first_timestamp = timestamps[0]
                last_timestamp = timestamps[-1]
                mid_point = len(timestamps) // 2
                
                # Use middle 20% of the timeline for a reasonable sample
                start_idx = max(0, mid_point - len(timestamps) // 5)
                end_idx = min(len(timestamps) - 1, mid_point + len(timestamps) // 5)
                
                start_time = str(timestamps[start_idx]) if hasattr(timestamps[start_idx], 'timestamp') else str(timestamps[start_idx])
                end_time = str(timestamps[end_idx]) if hasattr(timestamps[end_idx], 'timestamp') else str(timestamps[end_idx])
                
                print(f"DEBUG: Using default time range: {start_time} to {end_time}")
            else:
                return "Not enough timestamp data for time range analysis."
        
        # Convert time strings to appropriate formats
        try:
            # Try different formats for flexibility
            try:
                # Try interpreting as float/timestamp directly
                start = float(start_time)
                end = float(end_time)
            except ValueError:
                # Try parsing as ISO format or other string format
                try:
                    from dateutil import parser
                    start = parser.parse(start_time).timestamp()
                    end = parser.parse(end_time).timestamp()
                except:
                    # Last resort - just use string as is
                    start = start_time
                    end = end_time
            
            # Get the data and filter by time range
            result = self.analyzer.analyze_for_query(f"analyze {parameter} between {start} and {end}")
            
            # Format the response
            if parameter in result:
                data = result[parameter]
                summary = f"{parameter.capitalize()} analysis from {start} to {end}:\n"
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        summary += f"- {key.capitalize()}: {value}\n"
                elif isinstance(data, list):
                    summary += "Values: " + ", ".join(map(str, data[:5]))
                    if len(data) > 5:
                        summary += f" (and {len(data)-5} more values)"
                else:
                    summary += f"Value: {data}"
                
                return summary
            else:
                return f"No data available for {parameter} in the specified time range."
            
        except Exception as e:
            error_msg = f"Error analyzing time range: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg
    
    def _correlate_parameters(self, parameters: List[str]) -> str:
        """Find correlations between multiple telemetry parameters."""
        try:
            if len(parameters) < 2:
                return "Need at least two parameters to calculate correlation."
            
            # Check if any parameters are invalid or too generic
            valid_params = []
            for param in parameters:
                if param in ["gps", "altitude", "speed", "battery"]:
                    # These are too generic, try to find specific fields
                    specific_fields = []
                    for field in self.analyzer.time_series.keys():
                        if field != "timestamp" and param in field.lower():
                            specific_fields.append(field)
                            if len(specific_fields) >= 2:  # Limit to avoid too many
                                break
                    
                    if specific_fields:
                        valid_params.extend(specific_fields)
                        print(f"DEBUG: Replaced generic '{param}' with specific fields: {specific_fields}")
                    else:
                        valid_params.append(param)
                else:
                    valid_params.append(param)
            
            # Format the query for the analyzer with valid parameters
            param_str = " and ".join(valid_params)
            result = self.analyzer.analyze_for_query(f"correlate {param_str}")
            
            if "correlations" in result and result["correlations"]:
                correlations = result["correlations"]
                
                summary = "Parameter correlations:\n"
                for pair, corr in correlations.items():
                    summary += f"- {pair}: {corr:.3f}\n"
                return summary
            else:
                # Fallback to a simpler analysis if correlations aren't available
                summaries = []
                for param in valid_params[:3]:  # Limit to first 3 to avoid overload
                    try:
                        param_result = self.analyzer.analyze_for_query(f"analyze {param}")
                        if param in param_result:
                            summaries.append(f"{param}: {str(param_result[param])[:200]}...")
                    except Exception as e:
                        print(f"DEBUG: Error analyzing parameter {param}: {str(e)}")
                
                if summaries:
                    return "Parameter summaries (correlation not available):\n" + "\n".join(summaries)
                else:
                    return "No data available for the specified parameters. Try using more specific parameter names from the telemetry data."
        except Exception as e:
            error_msg = f"Error correlating parameters: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg
    
    def _execute_data_query(self, query: str) -> str:
        """Execute a custom data analysis query."""
        result = self.analyzer.analyze_for_query(query)
        
        # Convert complex objects to strings
        return json.dumps(result, indent=2, default=str)
    
    # Helper methods for parsing
    def _parse_thinking(self, thinking: Any) -> str:
        """Parse the thinking response."""
        if hasattr(thinking, "content"):
            return thinking.content
        return str(thinking)
    
    def _parse_plan(self, plan_response: Any) -> List[str]:
        """Parse the planning response into a list of steps."""
        if hasattr(plan_response, "content"):
            content = plan_response.content
        else:
            content = str(plan_response)
            
        # Extract numbered steps from the content
        plan_steps = []
        
        for line in content.split("\n"):
            line = line.strip()
            # Look for lines that start with a number followed by a period or parenthesis
            if line and (line[0].isdigit() or (line.startswith("Step") and ":" in line)):
                # Extract the step content
                if ":" in line:
                    step = line.split(":", 1)[1].strip()
                else:
                    # Find the first period or space after the number
                    for i, char in enumerate(line):
                        if char in ".) ":
                            step = line[i+1:].strip()
                            break
                    else:
                        step = line
                
                if step:
                    plan_steps.append(step)
        
        # If no steps were found, try to parse the whole response as a single step
        if not plan_steps and content.strip():
            plan_steps = [content.strip()]
            
        return plan_steps
    
    def _summarize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summarized version of analysis results for memory storage."""
        summary = {}
        
        # Summarize each tool's results
        for tool, results in analysis.items():
            if isinstance(results, list):
                summary[tool] = f"{len(results)} results from {tool}"
            else:
                summary[tool] = "Results available"
                
        return summary
    
    async def _async_run_workflow(self, initial_state: AgentState) -> Dict[str, Any]:
        """Asynchronously run the workflow."""
        try:
            # Create a configuration with the memory saver and recursion limit
            config = RunnableConfig(
                configurable={
                    "checkpoint_saver": self.memory_saver,  # For checkpointing
                },
                recursion_limit=50  # Increase from default 25
            )
            
            # Create the workflow compiler
            compiler = self.workflow.compile()
            
            # Run the workflow
            result = await compiler.ainvoke(
                initial_state,
                config=config
            )
            
            return result
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Error running workflow: {str(e)}\n{traceback_str}")
            
            # Return a simplified error state
            return {
                "query": initial_state["query"],
                "thoughts": initial_state.get("thoughts", []) + [f"Error: {str(e)}"],
                "answer": f"I encountered an error while analyzing the flight data: {str(e)}",
                "status": "ERROR"
            }
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory_manager.clear() 