#!/usr/bin/env python3
"""
FastMCP Server with Tavily Web Search Integration using Official SDK (Synchronous)

This MCP server provides web search capabilities using Tavily's official Python SDK.
It exposes search tools that can be used by MCP clients to perform web searches.
"""

import os
from typing import Any, Dict, List, Optional
from tavily import TavilyClient
from fastmcp import FastMCP
from openai import OpenAI
from pinecone import Pinecone
import requests
import json
from dotenv import load_dotenv
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("SOPS-AI")

# Tavily API configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Pinecone and OpenAI configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = "text-embedding-3-small"

# Request API configuration
REQUEST_SERVER_URL = os.getenv("REQUEST_SERVER_URL")
REQUEST_ACCESS_TOKEN = os.getenv("REQUEST_ACCESS_TOKEN")

# Initialize clients
tavily_client = None
openai_client = None
pinecone_index = None

if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    print("Warning: TAVILY_API_KEY environment variable not set")

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("Warning: OPENAI_API_KEY environment variable not set")

if PINECONE_API_KEY and PINECONE_INDEX_NAME:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        print(f"Warning: Could not connect to Pinecone: {e}")
else:
    print("Warning: PINECONE_API_KEY or PINECONE_INDEX_NAME environment variable not set")

if not REQUEST_SERVER_URL or not REQUEST_ACCESS_TOKEN:
    print("Warning: REQUEST_SERVER_URL or REQUEST_ACCESS_TOKEN environment variable not set")


@mcp.tool()
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True,
    include_raw_content: bool = False,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_images: bool = False
) -> Dict[str, Any]:
    """
    Search the web using Tavily's search API with the official SDK.
    
    Args:
        query: The search query string
        max_results: Maximum number of search results to return (default: 5)
        search_depth: Search depth - "basic" or "advanced" (default: "basic")
        include_answer: Whether to include a direct answer (default: True)
        include_raw_content: Whether to include raw content from pages (default: False)
        include_domains: List of domains to include in search
        exclude_domains: List of domains to exclude from search
        include_images: Whether to include images in results (default: False)
    
    Returns:
        Dictionary containing search results with answer, results, and metadata
    """
    
    if not tavily_client:
        return {
            "error": "Tavily client not initialized. Please set TAVILY_API_KEY environment variable."
        }
    
    try:
        # Use the SDK's search method (synchronous)
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_images=include_images
        )
        
        # Format the response for better readability
        formatted_response = {
            "query": query,
            "answer": response.get("answer", ""),
            "results": []
        }
        
        # Process search results
        for result in response.get("results", []):
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0)
            }
            
            # Include raw content if requested and available
            if include_raw_content and "raw_content" in result:
                formatted_result["raw_content"] = result["raw_content"]
            
            formatted_response["results"].append(formatted_result)
        
        # Add images if requested and available
        if include_images and "images" in response:
            formatted_response["images"] = response["images"]
        
        # Add metadata
        formatted_response["metadata"] = {
            "total_results": len(formatted_response["results"]),
            "search_depth": search_depth,
            "response_time": response.get("response_time", 0)
        }
        
        return formatted_response
        
    except Exception as e:
        return {
            "error": f"Search error: {str(e)}"
        }


@mcp.tool()
def kb_search(
    query: str,
    top_k: int = 5,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Search the knowledge base in Pinecone using semantic similarity.
    
    Args:
        query: The search query string
        top_k: Number of top similar chunks to return (default: 5)
        include_metadata: Whether to include metadata in results (default: True)
    
    Returns:
        Dictionary containing search results with similarity scores and metadata
    """
    
    if not pinecone_index:
        return {
            "error": "Pinecone index not initialized. Please check PINECONE_API_KEY and PINECONE_INDEX_NAME environment variables."
        }
    
    if not openai_client:
        return {
            "error": "OpenAI client not initialized. Please set OPENAI_API_KEY environment variable."
        }
    
    try:
        # Generate embedding for the query using OpenAI
        embedding_response = openai_client.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Search Pinecone for similar chunks
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata
        )
        
        # Format the response
        formatted_response = {
            "query": query,
            "results": []
        }
        
        for match in search_response.matches:
            result = {
                "id": match.id,
                "score": float(match.score),
                "text": match.metadata.get("text", "") if match.metadata else "",
                "source": match.metadata.get("source", "") if match.metadata else ""
            }
            
            # Include full metadata if requested
            if include_metadata and match.metadata:
                result["metadata"] = match.metadata
            
            formatted_response["results"].append(result)
        
        # Add search metadata
        formatted_response["metadata"] = {
            "total_results": len(formatted_response["results"]),
            "embedding_model": EMBEDDING_MODEL,
            "index_name": PINECONE_INDEX_NAME
        }
        
        return formatted_response
        
    except Exception as e:
        return {
            "error": f"Knowledge base search error: {str(e)}"
        }


@mcp.tool()
def create_request(
    subject: str,
    requester_email: str,
    category_name: str = "Request",
    cc_email_set: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    impact_name: str = "Low",
    priority_name: str = "Low",
    urgency_name: str = "Low",
    department_name: Optional[str] = None,
    location_name: Optional[str] = None,
    support_level: str = "Tier1",
    spam: bool = False,
    assignee_email: Optional[str] = None,
    technician_group_name: Optional[str] = None,
    source: str = "External",
    status_name: str = "Open",
    description: Optional[str] = None,
    custom_field: Optional[Dict[str, Any]] = None,
    link_asset_ids: Optional[List[Dict[str, Any]]] = None,
    link_ci_ids: Optional[List[Dict[str, Any]]] = None,
    file_attachments: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Create a new request/ticket in the system.
    
    Args:
        subject: Subject of the ticket (required)
        requester_email: Email address of the user registered for the client (required)
        category_name: Category Name of a Request (default: "Request")
        cc_email_set: Email addresses for notifications
        tags: Additional identifiers attached to a ticket
        impact_name: Effect of the Request - Low, On User, On department, Or On Business (default: "Low")
        priority_name: Importance of the Request - Low, Medium, High, or Urgent (default: "Low")
        urgency_name: Urgency level - Low, Medium, High, or Urgent (default: "Low")
        department_name: Department information
        location_name: Location where Request happened
        support_level: Level of support - Tier1, Tier2, Tier3, or Tier4 (default: "Tier1")
        spam: Whether request is spam (default: False)
        assignee_email: Email of the assignee
        technician_group_name: Name of the technician group
        source: Origin of the ticket (default: "External")
        status_name: Status - Open, In Progress, Pending, Resolved, Closed (default: "Open")
        description: Additional description of the request
        custom_field: Custom field key-value pairs
        link_asset_ids: Asset IDs to link - [{"assetModel": "asset_hardware", "assetId": 1}]
        link_ci_ids: CI IDs to link - [{"ciId": 2, "ciModel": "cmdb"}]
        file_attachments: File attachments - [{"refFileName": "abc", "realName": "xyz.pdf"}]
    
    Returns:
        Dictionary containing the created request details or error information
    """
    
    if not REQUEST_SERVER_URL or not REQUEST_ACCESS_TOKEN:
        return {
            "error": "Request API not configured. Please set REQUEST_SERVER_URL and REQUEST_ACCESS_TOKEN environment variables."
        }
    
    # Validate required fields
    if not subject or not subject.strip():
        return {
            "error": "Subject is required and cannot be empty."
        }
    
    if not requester_email or not requester_email.strip():
        return {
            "error": "Requester email is required and cannot be empty."
        }
    
    # Basic email validation
    if "@" not in requester_email or "." not in requester_email:
        return {
            "error": "Invalid requester email format."
        }
    
    # Validate enum values (case-insensitive for support level)
    valid_impact_names = ["Low", "On User", "On department", "Or On Business"]
    if impact_name not in valid_impact_names:
        return {
            "error": f"Invalid impact_name. Must be one of: {', '.join(valid_impact_names)}"
        }
    
    valid_priority_names = ["Low", "Medium", "High", "Urgent"]
    if priority_name not in valid_priority_names:
        return {
            "error": f"Invalid priority_name. Must be one of: {', '.join(valid_priority_names)}"
        }
    
    valid_urgency_names = ["Low", "Medium", "High", "Urgent"]
    if urgency_name not in valid_urgency_names:
        return {
            "error": f"Invalid urgency_name. Must be one of: {', '.join(valid_urgency_names)}"
        }
    
    # Support level validation - convert to lowercase for API
    valid_support_levels_api = ["tier1", "tier2", "tier3", "tier4"]
    support_level_lower = support_level.lower()
    if support_level_lower not in valid_support_levels_api:
        return {
            "error": "Invalid support_level. Must be one of: Tier1, Tier2, Tier3, Tier4"
        }
    
    valid_status_names = ["Open", "In Progress", "Pending", "Resolved", "Closed"]
    if status_name not in valid_status_names:
        return {
            "error": f"Invalid status_name. Must be one of: {', '.join(valid_status_names)}"
        }
    
    # Prepare the request payload matching the documentation format
    payload = {
        "subject": subject.strip(),
        "requesterEmail": requester_email.strip(),
        "impactName": impact_name,
        "priorityName": priority_name,
        "urgencyName": urgency_name,
        "statusName": status_name,
        "spam": spam
    }
    
    # Add optional fields following documentation format
    if category_name and category_name != "Request":
        payload["categoryName"] = category_name
    
    # Use lowercase for support level as shown in documentation
    if support_level:
        payload["supportLevel"] = support_level.lower()
    
    # Add source only if it's not the default
    if source and source != "External":
        payload["source"] = source
    
    # Add optional fields if provided
    if cc_email_set:
        # Validate CC emails
        for email in cc_email_set:
            if "@" not in email or "." not in email:
                return {
                    "error": f"Invalid CC email format: {email}"
                }
        payload["ccEmailSet"] = cc_email_set
    
    if tags:
        payload["tags"] = tags
    
    if department_name:
        payload["departmentName"] = department_name.strip()
    
    if location_name:
        payload["locationName"] = location_name.strip()
    
    if assignee_email:
        if "@" not in assignee_email or "." not in assignee_email:
            return {
                "error": "Invalid assignee email format."
            }
        payload["assigneeEmail"] = assignee_email.strip()
    
    if technician_group_name:
        payload["technicianGroupName"] = technician_group_name.strip()
    
    if description:
        payload["description"] = description.strip()
    
    if custom_field:
        payload["customField"] = custom_field
    
    if link_asset_ids:
        payload["linkAssetIds"] = link_asset_ids
    
    if link_ci_ids:
        payload["linkCiIds"] = link_ci_ids
    
    if file_attachments:
        payload["fileAttachments"] = file_attachments
    
    # Prepare headers with correct Bearer token format
    headers = {
        "Authorization": f"Bearer {REQUEST_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Make the API request
    try:
        url = f"{REQUEST_SERVER_URL.rstrip('/')}/api/v1/request"
        
        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Handle different response status codes
        if response.status_code == 200 or response.status_code == 201:
            try:
                response_data = response.json()
                return {
                    "success": True,
                    "message": "Request created successfully",
                    "request_data": response_data
                }
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "message": "Request created successfully",
                    "raw_response": response.text
                }
        
        elif response.status_code == 400:
            try:
                error_data = response.json()
                return {
                    "error": f"Bad Request: {error_data.get('message', 'Invalid request data')}",
                    "details": error_data
                }
            except json.JSONDecodeError:
                return {
                    "error": f"Bad Request: {response.text}"
                }
        
        elif response.status_code == 401:
            return {
                "error": "Unauthorized: Invalid or expired access token. Please check REQUEST_ACCESS_TOKEN."
            }
        
        elif response.status_code == 403:
            return {
                "error": "Forbidden: You don't have permission to create requests."
            }
        
        elif response.status_code == 404:
            return {
                "error": "Not Found: The API endpoint was not found. Please check REQUEST_SERVER_URL."
            }
        
        elif response.status_code == 500:
            return {
                "error": "Internal Server Error: The server encountered an error while processing the request."
            }
        
        else:
            try:
                error_data = response.json()
                return {
                    "error": f"API request failed with status {response.status_code}",
                    "details": error_data
                }
            except json.JSONDecodeError:
                return {
                    "error": f"API request failed with status {response.status_code}: {response.text}"
                }
    
    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout: The API request took too long to complete. Please try again."
        }
    
    except requests.exceptions.ConnectionError:
        return {
            "error": f"Connection error: Could not connect to {REQUEST_SERVER_URL}. Please check the server URL."
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Request error: {str(e)}"
        }
    
    except Exception as e:
        return {
            "error": f"Unexpected error while creating request: {str(e)}"
        }


def main():
    """Main server startup function."""
    print("🔍 Starting Tavily Web Search MCP Server (Official SDK)...")
    print(f"📡 Server name: {mcp.name}")
    print("🔧 Available tools: web_search, kb_search, create_request")
    if TAVILY_API_KEY:
        print("✅ Tavily API key configured")
    else:
        print("❌ Warning: Tavily API key not found")
    
    if OPENAI_API_KEY:
        print("✅ OpenAI API key configured")
    else:
        print("❌ Warning: OpenAI API key not found")
    
    if pinecone_index:
        print("✅ Pinecone knowledge base connected")
    else:
        print("❌ Warning: Pinecone knowledge base not available")
    
    if REQUEST_SERVER_URL and REQUEST_ACCESS_TOKEN:
        print("✅ Request API configured")
    else:
        print("❌ Warning: Request API not configured")
    
    # Run the server (synchronous)
    mcp.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Shutting down Tavily Web Search MCP Server...")
    except Exception as e:
        print(f"❌ Server error: {e}")