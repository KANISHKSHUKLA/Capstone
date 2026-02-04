from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import logging
import uuid
import os
from typing import Dict, Any

from app.core.paper_analyzer import PaperAnalyzer
from app.core.chat_manager import ChatManager
from app.clients.gemini_ai import GeminiClient
from app.models.schemas import ErrorResponse, ChatRequest
from app.storage.json_storage import JSONStorage

router = APIRouter()

# In-memory cache for active processing jobs
analysis_cache: Dict[str, Dict[str, Any]] = {}

# Initialize JSON storage
json_storage = JSONStorage()

@router.post("/analyze", 
             response_model=Dict[str, Any],
             responses={
                 200: {"description": "Analysis job started successfully"},
                 400: {"model": ErrorResponse, "description": "Bad request"},
                 415: {"model": ErrorResponse, "description": "Unsupported file type"}
             })
async def analyze_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    paper_analyzer: PaperAnalyzer = Depends(lambda: PaperAnalyzer())
):
    """
    Endpoint to submit a research paper for analysis.
    Returns a job ID immediately and processes the paper in the background.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported."
        )
    
    job_id = str(uuid.uuid4())
    
    analysis_cache[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "result": None
    }
    
    file_content = await file.read()
    
    async def process_paper(job_id: str, filename: str, file_content: bytes):
        try:
            temp_file_path = await paper_analyzer.save_upload_file(file_content)
            
            result = await paper_analyzer.analyze_paper(temp_file_path)
            
            analysis_cache[job_id]["status"] = "completed"
            analysis_cache[job_id]["result"] = result
            
            # Save to JSON storage when all 5 calls succeed
            try:
                save_data = {
                    "filename": filename,
                    "result": result
                }
                json_storage.save_paper_analysis(job_id, save_data)
                logging.info(f"Successfully saved analysis to JSON for job: {job_id}")
            except Exception as save_error:
                logging.error(f"Failed to save analysis to JSON: {str(save_error)}")
            
        except Exception as e:
            logging.error(f"Error processing paper: {str(e)}")
            analysis_cache[job_id]["status"] = "failed"
            analysis_cache[job_id]["error"] = str(e)
    
    background_tasks.add_task(process_paper, job_id, file.filename, file_content)
    
    return {"job_id": job_id, "status": "processing"}

@router.get("/status/{job_id}", 
            response_model=Dict[str, Any],
            responses={
                404: {"model": ErrorResponse, "description": "Job not found"}
            })
async def get_job_status(job_id: str):
    """
    Check the status of a paper analysis job.
    If the job is complete, returns the analysis results.
    Loads from JSON storage if not in memory cache.
    """
    # Check in-memory cache first
    if job_id in analysis_cache:
        job_info = analysis_cache[job_id]
        
        if job_info["status"] == "completed":
            return {
                "status": "completed",
                "filename": job_info["filename"],
                "result": job_info["result"]
            }
        elif job_info["status"] == "failed":
            return {
                "status": "failed",
                "filename": job_info["filename"],
                "error": job_info.get("error", "Unknown error")
            }
        else:
            return {
                "status": "processing",
                "filename": job_info["filename"]
            }
    
    # If not in cache, try loading from JSON storage
    try:
        stored_data = json_storage.load_paper_analysis(job_id)
        if stored_data:
            # Load into cache for future requests
            analysis_cache[job_id] = {
                "status": stored_data.get("status", "completed"),
                "filename": stored_data.get("filename", "Unknown"),
                "result": stored_data.get("result", {})
            }
            
            return {
                "status": "completed",
                "filename": stored_data.get("filename", "Unknown"),
                "result": stored_data.get("result", {})
            }
    except Exception as e:
        logging.error(f"Error loading from JSON storage: {str(e)}")
    
    # Not found in cache or storage
    raise HTTPException(
        status_code=404,
        detail=f"Job ID {job_id} not found"
    )

@router.delete("/jobs/{job_id}", 
               responses={
                   200: {"description": "Job deleted successfully"},
                   404: {"model": ErrorResponse, "description": "Job not found"}
               })
async def delete_job(job_id: str):
    """
    Delete a job and its results from the cache and JSON storage.
    """
    found = False
    
    # Delete from memory cache
    if job_id in analysis_cache:
        try:
            del analysis_cache[job_id]
            found = True
        except Exception as e:
            logging.error(f"Error deleting from cache: {str(e)}")
    
    # Delete from JSON storage
    try:
        if json_storage.paper_exists(job_id):
            json_storage.delete_paper_analysis(job_id)
            json_storage.delete_chat_history(job_id)
            found = True
    except Exception as e:
        logging.error(f"Error deleting from JSON storage: {str(e)}")
    
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Job ID {job_id} not found"
        )
    
    return {"message": f"Job {job_id} deleted successfully"}


@router.post("/jobs/{job_id}/chat", 
             response_model=Dict[str, Any],
             responses={
                 200: {"description": "Chat response"},
                 404: {"model": ErrorResponse, "description": "Job not found"},
                 400: {"model": ErrorResponse, "description": "Bad request"}
             })
async def chat_with_paper(job_id: str, chat_request: ChatRequest):
    """
    Chat with an AI about a previously analyzed paper.
    Loads from JSON storage if not in memory cache.
    
    Args:
        job_id: The ID of the analysis job
        chat_request: The chat request containing the user's message
        
    Returns:
        Dictionary containing the response and updated chat history
    """
    job_info = None
    
    # Check in-memory cache first
    if job_id in analysis_cache:
        job_info = analysis_cache[job_id]
    else:
        # Try loading from JSON storage
        try:
            stored_data = json_storage.load_paper_analysis(job_id)
            if stored_data:
                job_info = {
                    "status": stored_data.get("status", "completed"),
                    "filename": stored_data.get("filename", "Unknown"),
                    "result": stored_data.get("result", {}),
                    "chat_history": json_storage.load_chat_history(job_id)
                }
                # Load into cache
                analysis_cache[job_id] = job_info
        except Exception as e:
            logging.error(f"Error loading from JSON storage: {str(e)}")
    
    if not job_info:
        raise HTTPException(
            status_code=404,
            detail=f"Job ID {job_id} not found"
        )
    
    if job_info["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed yet"
        )
    
    if "result" not in job_info or not job_info["result"]:
        raise HTTPException(
            status_code=400,
            detail=f"No analysis result found for job {job_id}"
        )
    
    if "chat_history" not in job_info:
        job_info["chat_history"] = []
    
    chat_manager = ChatManager(GeminiClient())
    
    try:
        chat_result = await chat_manager.process_chat_message(
            job_id=job_id,
            user_message=chat_request.message,
            analysis_result=job_info["result"],
            chat_history=job_info["chat_history"]
        )
        
        job_info["chat_history"] = chat_result["updated_history"]
        
        # Save chat history to JSON storage
        try:
            json_storage.save_chat_history(job_id, chat_result["updated_history"])
            logging.info(f"Successfully saved chat history to JSON for job: {job_id}")
        except Exception as save_error:
            logging.error(f"Failed to save chat history to JSON: {str(save_error)}")
        
        return {
            "response": chat_result["response"],
            "job_id": job_id
        }
    except Exception as e:
        logging.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat message: {str(e)}"
        )


@router.get("/jobs/{job_id}/chat", 
            response_model=Dict[str, Any],
            responses={
                200: {"description": "Chat history"},
                404: {"model": ErrorResponse, "description": "Job not found"}
            })
async def get_chat_history(job_id: str):
    """
    Get the chat history for a specific job.
    Loads from JSON storage if not in memory cache.
    
    Args:
        job_id: The ID of the analysis job
        
    Returns:
        Dictionary containing the chat history
    """
    chat_history = []
    
    # Check in-memory cache first
    if job_id in analysis_cache:
        job_info = analysis_cache[job_id]
        chat_history = job_info.get("chat_history", [])
        # If cache has no chat history yet, try loading from storage and hydrate cache
        if not chat_history:
            try:
                loaded_history = json_storage.load_chat_history(job_id)
                if loaded_history:
                    chat_history = loaded_history
                    job_info["chat_history"] = loaded_history
                    analysis_cache[job_id] = job_info
            except Exception as e:
                logging.error(f"Error hydrating chat history from storage: {str(e)}")
    else:
        # Try loading from JSON storage
        try:
            if json_storage.paper_exists(job_id):
                chat_history = json_storage.load_chat_history(job_id)
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Job ID {job_id} not found"
                )
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error loading chat history from JSON storage: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=f"Job ID {job_id} not found"
            )
    
    return {
        "job_id": job_id,
        "chat_history": chat_history
    }
