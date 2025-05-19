from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import uuid
import asyncio
import re  # For regex pattern matching

from agents.flight_agent import FlightAgent
from agents.react_agent import ReActAgent
from telemetry.parser import TelemetryParser
from telemetry.analyzer import TelemetryAnalyzer

# Load environment variables
load_dotenv()

# Check if OPENAI_API_KEY is set and provide clear error if not
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY environment variable not found! Using demo key for testing.")
    print("For production use, please set this in your .env file or environment variables.")
    # Set a temporary key for testing - REPLACE THIS WITH YOUR ACTUAL KEY
    os.environ["OPENAI_API_KEY"] = "sk-1234567890abcdefghijklmnopqrstuvwxyz"
    
# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# CORS configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",")
ALLOWED_METHODS = os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE").split(",")
ALLOWED_HEADERS = os.getenv("ALLOWED_HEADERS", "Content-Type,Accept,Origin,X-Requested-With").split(",")
CORS_MAX_AGE = int(os.getenv("CORS_MAX_AGE", "3600"))

# File upload configuration
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "temp")

app = FastAPI()

# Add CORS middleware with explicit method and header restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
    expose_headers=["*"],  # Headers that can be exposed to the browser
    max_age=CORS_MAX_AGE,  # Maximum time to cache pre-flight requests (in seconds)
)

# In-memory storage
flight_sessions = {}
active_sessions = {}

class FlightSession(BaseModel):
    id: str
    created_at: datetime
    telemetry_data: Dict
    agent_type: str = "react_agent"

class ChatMessage(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    analysis: Optional[Dict[str, Any]] = None
    thought_process: Optional[List[str]] = None
    tools_used: Optional[List[str]] = None

@app.post("/upload")
async def upload_log(
    file: UploadFile = File(...),
    agent_type: str = Form("react_agent")  # Default to original agent for backward compatibility
):
    try:
        # Validate agent type
        if agent_type not in ["flight_agent", "react_agent"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid agent_type. Must be one of: flight_agent, react_agent"
            )
        
        # Create temporary file
        file_path = f"./backend/temp/{uuid.uuid4()}_{file.filename}"
        os.makedirs("./backend/temp", exist_ok=True)
        
        # Write uploaded file to disk
        try:
            with open(file_path, "wb") as f:
                while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                    f.write(content)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            print(f"ERROR writing uploaded file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to write file: {str(e)}")
        
        # Process the uploaded file
        try:
            print(f"Processing file: {file.filename}")
            parser = TelemetryParser(file_path)
            telemetry_data = parser.parse()
            if not telemetry_data:
                raise ValueError("No telemetry data could be extracted from the log file.")
            
            print(f"Creating telemetry analyzer...")
            analyzer = TelemetryAnalyzer(telemetry_data)
            
            # Create new flight session
            session_id = str(uuid.uuid4())
            
            # Store session info
            flight_sessions[session_id] = FlightSession(
                id=session_id,
                created_at=datetime.now(timezone.utc),
                telemetry_data=telemetry_data,
                agent_type=agent_type
            )
            
            # Create the appropriate agent type
            print(f"Creating {agent_type} for session {session_id}...")
            
            if agent_type == "react_agent":
                # Create ReAct agent
                active_sessions[session_id] = ReActAgent(
                    session_id=session_id,
                    telemetry_data=telemetry_data,
                    analyzer=analyzer
                )
            else:
                # Create the original flight agent
                active_sessions[session_id] = FlightAgent(
                    session_id=session_id,
                    telemetry_data=telemetry_data,
                    analyzer=analyzer
                )
            
            # Cleanup
            os.remove(file_path)
            
            print(f"Successfully created session {session_id} from {file.filename}")
            return {"session_id": session_id, "message": "Log file processed successfully", "agent_type": agent_type}
        
        except Exception as processing_error:
            # If file exists but processing fails, clean up the file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Log the error with stack trace
            print(f"ERROR processing log file: {str(processing_error)}")
            import traceback
            print(f"UPLOAD PROCESSING ERROR: {traceback.format_exc()}")
            
            # Return specific error for API clients
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to process log file: {str(processing_error)}"
            )
    
    except Exception as e:
        # Log the error with stack trace
        print(f"ERROR in upload endpoint: {str(e)}")
        import traceback
        print(f"UPLOAD ENDPOINT ERROR: {traceback.format_exc()}")
        
        # Return specific error for API clients
        raise HTTPException(
            status_code=400, 
            detail=f"Upload failed: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    session_id = message.session_id
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Process message with agent with a hard timeout at API level
        try:
            # Set an overall timeout for the chat operation
            result = await asyncio.wait_for(
                active_sessions[session_id].process_message(message.message),
                timeout=40.0  # 40 seconds max at API level
            )
        except asyncio.TimeoutError:
            print(f"ERROR: Chat endpoint timed out after 40 seconds for session {session_id}")
            raise HTTPException(
                status_code=504, 
                detail="The request took too long to process. Please try a simpler query."
            )
        
        # Check if result contains an error field, which indicates a failure
        if "error" in result and result["error"]:
            print(f"ERROR returned from agent: {result['error']}")
            
            # Provide the error message to the client
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing chat: {result['error']}"
            )
        
        # Build analysis data from the result
        analysis_data = {
            "type": "Flight Analysis",
            "metrics": {},
            "anomalies": "No anomalies detected"
        }
        
        # Extract metrics from result
        if "analysis" in result and isinstance(result["analysis"], dict):
            # Extract metrics data, prioritizing proper altitude metrics
            metrics_data = result["analysis"].get("metrics", {})
            
            # Look for altitude analysis in the result
            altitude_analysis = None
            if "altitude_analysis" in result["analysis"]:
                altitude_analysis = result["analysis"]["altitude_analysis"]
                # Store the altitude analysis in metrics explicitly
                if altitude_analysis and "statistics" in altitude_analysis:
                    metrics_data["altitude"] = altitude_analysis["statistics"]
                    # Also record if this was converted from absolute altitude
                    if "is_absolute_altitude" in altitude_analysis:
                        metrics_data["altitude"]["source_was_absolute"] = altitude_analysis["is_absolute_altitude"]
                    if "field_used" in altitude_analysis:
                        metrics_data["altitude"]["field_used"] = altitude_analysis["field_used"]
            elif "altitude" in metrics_data:
                altitude_analysis = {"statistics": metrics_data["altitude"]}
            
            # If we have altitude analysis, make sure it's using reasonable values
            if altitude_analysis and "statistics" in altitude_analysis:
                alt_stats = altitude_analysis["statistics"]
                # Check if the max altitude is reasonable
                max_alt = alt_stats.get("max")
                if max_alt is not None and isinstance(max_alt, (int, float)) and max_alt > 1000:
                    print(f"WARNING: Unreasonably high max altitude in API response: {max_alt}")
                    # Try to apply an additional correction factor if it looks like sea level data
                    if max_alt > 100000:  # Extremely high, might be in mm or µm
                        alt_stats["max"] = max_alt / 1000.0
                        print(f"Applied mm->m conversion: {alt_stats['max']}")
                    else:
                        # Otherwise, flag this value as suspicious
                        alt_stats["max"] = f"Suspicious value: {max_alt}"
                
                # Verify min altitude is present
                min_alt = alt_stats.get("min")
                if min_alt is None:
                    print(f"WARNING: Missing min altitude in API response, attempting to calculate")
                    # If missing, try to derive from range
                    if "range" in alt_stats and max_alt is not None and isinstance(max_alt, (int, float)):
                        range_value = alt_stats.get("range")
                        if isinstance(range_value, (int, float)):
                            alt_stats["min"] = max_alt - range_value
                            print(f"Calculated min altitude: {alt_stats['min']}")
                
                # Use the verified altitude stats
                metrics_data["altitude"] = alt_stats
        
            # Include up to 30 metrics fields maximum to avoid overwhelming response
            field_count = 0
            pruned_metrics = {}
            
            # First, add any altitude-related metrics
            for field_name, field_data in metrics_data.items():
                if any(term in field_name.lower() for term in ["alt", "height"]):
                    pruned_metrics[field_name] = field_data
                    field_count += 1
            
            # Then add other important metrics
            for field_name, field_data in metrics_data.items():
                if field_count >= 30:
                    break
                    
                if field_name not in pruned_metrics and any(term in field_name.lower() for term in 
                                                           ["speed", "battery", "volt", "gps", "position"]):
                    pruned_metrics[field_name] = field_data
                    field_count += 1
            
            # Finally add remaining metrics up to the limit
            for field_name, field_data in metrics_data.items():
                if field_count >= 30:
                    break
                    
                if field_name not in pruned_metrics:
                    pruned_metrics[field_name] = field_data
                    field_count += 1
            
            analysis_data["metrics"] = pruned_metrics
            
            # Extract anomalies data
            anomalies_data = result["analysis"].get("anomalies", [])
            if isinstance(anomalies_data, list) and anomalies_data:
                analysis_data["anomalies"] = f"Detected {len(anomalies_data)} anomalies"
            elif isinstance(anomalies_data, str):
                analysis_data["anomalies"] = anomalies_data
        
        # Extract response content based on agent type
        session = flight_sessions.get(session_id)
        response_text = result.get("response", result.get("answer", ""))
        
        # ReAct agent specific fields
        thought_process = None
        tools_used = None
        
        # Get agent-specific data
        if session and session.agent_type == "react_agent":
            # For ReAct agent, include the thought process and tools used
            thought_process = result.get("thought_process", [])
            tools_used = result.get("tools_used", [])
        else:
            # For the original flight agent, correct known issues in the response
            # CRITICAL: Check LLM response for unreasonable altitude values and correct them
            if "altitude" in analysis_data["metrics"] and "min" in analysis_data["metrics"]["altitude"]:
                min_alt_value = analysis_data["metrics"]["altitude"]["min"]
                if isinstance(min_alt_value, (int, float)):
                    min_alt_str = f"{min_alt_value:.1f}" if min_alt_value != int(min_alt_value) else f"{int(min_alt_value)}"
                    # Replace incorrect statements about min altitude not being available
                    incorrect_patterns = [
                        r"(?:does not explicitly state|doesn't include|doesn't show|no data for|missing|unavailable) (?:the )?minimum altitude",
                        r"minimum altitude (?:is not available|is missing|was not provided|isn't included|isn't given)",
                        r"not (?:the|a) minimum altitude",
                        r"without the minimum (?:altitude|value)"
                    ]
                    for pattern in incorrect_patterns:
                        response_text = re.sub(
                            pattern, 
                            f"minimum altitude was {min_alt_str} m", 
                            response_text, 
                            flags=re.IGNORECASE
                        )
        
        return ChatResponse(
            response=response_text, 
            analysis=analysis_data,
            thought_process=thought_process,
            tools_used=tools_used
        )
    
    except Exception as e:
        # Log the error with stack trace
        print(f"ERROR in chat endpoint: {str(e)}")
        import traceback
        print(f"CHAT ENDPOINT ERROR: {traceback.format_exc()}")
        
        # Return specific error for API clients
        raise HTTPException(
            status_code=500, 
            detail=f"Chat processing failed: {str(e)}"
        )

@app.get("/session/{session_id}/messages")
async def get_session_messages(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get session messages using memory manager
    messages = active_sessions[session_id].memory_manager.get_session_messages()
    
    return {"messages": messages}

@app.get("/sessions")
async def list_sessions():
    # Return all available flight sessions
    return {
        "sessions": [
            {
                "id": session_id,
                "created_at": session.created_at.isoformat(),
                "has_telemetry": bool(session.telemetry_data),
                "agent_type": session.agent_type
            } for session_id, session in flight_sessions.items()
        ]
    }

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clean up session resources
    if session_id in active_sessions:
        active_sessions[session_id].clear_memory()
        del active_sessions[session_id]
    
    return {"message": "Session ended successfully"}

@app.get("/")
async def root():
    return {"status": "API is running", "endpoints": ["/upload", "/chat", "/sessions", "/session/{session_id}/messages"]}

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True) 