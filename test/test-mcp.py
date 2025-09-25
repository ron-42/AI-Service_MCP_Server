#!/usr/bin/env python3
"""
Direct test script for SOPS-AI MCP Server functions

This script tests the underlying functions without going through FastMCP decorators
"""

import os
import sys
import json
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Add the src directory to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)

load_dotenv()

# Import the clients and configurations directly from main
try:
    from main import (
        tavily_client, openai_client, pinecone_index,
        REQUEST_SERVER_URL, REQUEST_ACCESS_TOKEN, EMBEDDING_MODEL, PINECONE_INDEX_NAME
    )
    print("✅ Successfully imported MCP server components")
except ImportError as e:
    print(f"❌ Error importing components: {e}")
    sys.exit(1)

def test_web_search_direct(query: str, max_results: int = 5):
    """Test web search directly using the tavily client"""
    print(f"\n🔍 Testing Web Search: '{query}'")
    
    if not tavily_client:
        return {"error": "Tavily client not initialized"}
    
    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=True
        )
        
        formatted_response = {
            "query": query,
            "answer": response.get("answer", ""),
            "results": []
        }
        
        for result in response.get("results", []):
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0)
            }
            formatted_response["results"].append(formatted_result)
        
        return formatted_response
        
    except Exception as e:
        return {"error": f"Search error: {str(e)}"}


def test_kb_search_direct(query: str, top_k: int = 3):
    """Test KB search directly using openai and pinecone clients"""
    print(f"\n🧠 Testing KB Search: '{query}'")
    
    if not pinecone_index:
        return {"error": "Pinecone index not initialized"}
    
    if not openai_client:
        return {"error": "OpenAI client not initialized"}
    
    try:
        # Generate embedding
        embedding_response = openai_client.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Search Pinecone
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
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
            formatted_response["results"].append(result)
        
        return formatted_response
        
    except Exception as e:
        return {"error": f"KB search error: {str(e)}"}


def test_create_request_direct(subject: str, requester_email: str, **kwargs):
    """Test request creation directly using requests with correct Bearer token and payload format"""
    print(f"\n🎫 Testing Create Request: '{subject}'")
    
    if not REQUEST_SERVER_URL or not REQUEST_ACCESS_TOKEN:
        return {"error": "Request API not configured"}
    
    # Basic validation
    if not subject or not subject.strip():
        return {"error": "Subject is required"}
    
    if not requester_email or "@" not in requester_email:
        return {"error": "Valid requester email is required"}
    
    # Build payload matching the exact format from documentation
    payload = {
        "subject": subject.strip(),
        "requesterEmail": requester_email.strip(),
        "impactName": kwargs.get("impact_name", "Low"),
        "priorityName": kwargs.get("priority_name", "Low"),
        "urgencyName": kwargs.get("urgency_name", "Low"),
        "statusName": kwargs.get("status_name", "Open"),
        "spam": kwargs.get("spam", False)
    }
    
    # Add categoryName only if provided (documentation shows it's optional)
    if kwargs.get("category_name"):
        payload["categoryName"] = kwargs["category_name"]
    
    # Add supportLevel with correct case (documentation shows "tier2" not "Tier2")
    if kwargs.get("support_level"):
        payload["supportLevel"] = kwargs["support_level"].lower()
    
    # Add optional fields exactly as shown in documentation
    if kwargs.get("description"):
        payload["description"] = kwargs["description"]
    if kwargs.get("tags"):
        payload["tags"] = kwargs["tags"]
    if kwargs.get("department_name"):
        payload["departmentName"] = kwargs["department_name"]
    if kwargs.get("location_name"):
        payload["locationName"] = kwargs["location_name"]
    if kwargs.get("assignee_email"):
        payload["assigneeEmail"] = kwargs["assignee_email"]
    if kwargs.get("technician_group_name"):
        payload["technicianGroupName"] = kwargs["technician_group_name"]
    if kwargs.get("cc_email_set"):
        payload["ccEmailSet"] = kwargs["cc_email_set"]
    if kwargs.get("custom_field"):
        payload["customField"] = kwargs["custom_field"]
    if kwargs.get("link_asset_ids"):
        payload["linkAssetIds"] = kwargs["link_asset_ids"]
    if kwargs.get("link_ci_ids"):
        payload["linkCiIds"] = kwargs["link_ci_ids"]
    if kwargs.get("file_attachments"):
        payload["fileAttachments"] = kwargs["file_attachments"]
    
    # Use Bearer token with correct format
    headers = {
        "Authorization": f"Bearer {REQUEST_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        url = f"{REQUEST_SERVER_URL.rstrip('/')}/api/v1/request"
        print(f"Making request to: {url}")
        print(f"Headers: Authorization: Bearer {REQUEST_ACCESS_TOKEN[:20]}...")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url=url, headers=headers, json=payload, timeout=30)
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code in [200, 201]:
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
        else:
            try:
                error_data = response.json()
                return {
                    "error": f"API error ({response.status_code}): {error_data.get('message', error_data.get('userMessage', 'Unknown error'))}",
                    "details": error_data,
                    "raw_response": response.text
                }
            except json.JSONDecodeError:
                return {
                    "error": f"API error ({response.status_code}): {response.text}"
                }
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Request error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def test_minimal_request():
    """Test with minimal required fields only"""
    print("\n🔬 Testing Minimal Request (Only Required Fields)")
    
    if not REQUEST_SERVER_URL or not REQUEST_ACCESS_TOKEN:
        return {"error": "Request API not configured"}
    
    # Minimal payload matching documentation exactly
    payload = {
        "subject": "Minimal test request",
        "requesterEmail": "test@example.com"
    }
    
    headers = {
        "Authorization": f"Bearer {REQUEST_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        url = f"{REQUEST_SERVER_URL.rstrip('/')}/api/v1/request"
        print(f"Making minimal request to: {url}")
        print(f"Minimal Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url=url, headers=headers, json=payload, timeout=30)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code in [200, 201]:
            try:
                response_data = response.json()
                print("✅ Minimal Request Success!")
                return {
                    "success": True,
                    "message": "Minimal request created successfully",
                    "request_data": response_data
                }
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "message": "Minimal request created successfully",
                    "raw_response": response.text
                }
        else:
            try:
                error_data = response.json()
                print(f"❌ Minimal Request Failed: {error_data}")
                return {
                    "error": f"API error ({response.status_code}): {error_data.get('message', error_data.get('userMessage', 'Unknown error'))}",
                    "details": error_data
                }
            except json.JSONDecodeError:
                return {
                    "error": f"API error ({response.status_code}): {response.text}"
                }
    
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def run_all_tests():
    """Run comprehensive tests of all tools"""
    print("🧪 SOPS-AI MCP SERVER DIRECT TESTING")
    print("=" * 60)
    
    # Check environment
    env_vars = {
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
        "REQUEST_SERVER_URL": os.getenv("REQUEST_SERVER_URL"),
        "REQUEST_ACCESS_TOKEN": os.getenv("REQUEST_ACCESS_TOKEN")
    }
    
    print("Environment Check:")
    for var, value in env_vars.items():
        status = "✅" if value else "❌"
        print(f"{status} {var}: {'Set' if value else 'Not set'}")
    
    print("\n" + "=" * 60)
    
    # Test Web Search
    if tavily_client:
        result = test_web_search_direct("Python FastMCP tutorial", 2)
        if "error" in result:
            print(f"❌ Web Search Error: {result['error']}")
        else:
            print(f"✅ Web Search Success: Found {len(result['results'])} results")
            if result.get('answer'):
                print(f"   Answer: {result['answer'][:100]}...")
            if result.get('results'):
                print(f"   First result: {result['results'][0]['title']}")
    else:
        print("❌ Web Search: Tavily client not available")
    
    # Test KB Search  
    if openai_client and pinecone_index:
        result = test_kb_search_direct("API integration", 3)
        if "error" in result:
            print(f"❌ KB Search Error: {result['error']}")
        else:
            print(f"✅ KB Search Success: Found {len(result['results'])} results")
            for i, res in enumerate(result['results'][:2], 1):
                print(f"   {i}. Score: {res['score']:.3f} - {res['text'][:50]}...")
    else:
        print("❌ KB Search: OpenAI or Pinecone client not available")
    
    # Test Create Request - First try minimal request
    if REQUEST_SERVER_URL and REQUEST_ACCESS_TOKEN:
        print("\n" + "=" * 60)
        print("TESTING CREATE REQUEST FUNCTIONALITY")
        print("=" * 60)
        
        # Test 1: Minimal request
        minimal_result = test_minimal_request()
        if "error" in minimal_result:
            print(f"❌ Minimal Request Error: {minimal_result['error']}")
            if minimal_result.get('details'):
                print(f"   Details: {minimal_result['details']}")
        else:
            print(f"✅ Minimal Request Success: {minimal_result['message']}")
            if minimal_result.get('request_data'):
                req_data = minimal_result['request_data']
                print(f"   Request ID: {req_data.get('id', 'N/A')}")
                print(f"   Name: {req_data.get('name', 'N/A')}")
        
        # Test 2: Full request with all optional fields
        print("\n" + "-" * 40)
        result = test_create_request_direct(
            subject="Complete test request - All fields",
            requester_email="test@example.com",
            priority_name="Medium",
            urgency_name="Medium",
            impact_name="Low",
            status_name="Open",
            category_name="Network",
            support_level="tier2",  # Use lowercase as per documentation
            department_name="IT",
            assignee_email="admin@example.com",
            technician_group_name="IT Support",
            cc_email_set=["manager@example.com"],
            tags=["test", "mcp", "fixed-auth", "complete"],
            spam=False,
        )
        
        if "error" in result:
            print(f"❌ Full Create Request Error: {result['error']}")
            if result.get('details'):
                print(f"   Details: {result['details']}")
            if result.get('raw_response'):
                print(f"   Raw Response: {result['raw_response'][:200]}...")
        else:
            print(f"✅ Full Create Request Success: {result['message']}")
            if result.get('request_data'):
                req_data = result['request_data']
                print(f"   Request ID: {req_data.get('id', 'N/A')}")
                print(f"   Name: {req_data.get('name', 'N/A')}")
                print(f"   Created Time: {req_data.get('createdTime', 'N/A')}")
                print(f"   Support Level: {req_data.get('supportLevel', 'N/A')}")
    
    else:
        print("❌ Create Request: API configuration not available")
    
    print("\n" + "=" * 60)
    print("🎉 TESTING COMPLETED")
    print("=" * 60)
    print("Key improvements in this version:")
    print("- Fixed authentication header to use 'Bearer' format")
    print("- Corrected payload format to match API documentation")
    print("- Used lowercase support levels (tier1, tier2, etc.)")
    print("- Added minimal request test to isolate issues")
    print("- Enhanced error reporting with detailed debugging")
    print("- Added payload visualization for troubleshooting")
    print("\nIf you're still getting errors, the issue might be:")
    print("1. Invalid access token (expired or wrong scope)")
    print("2. Server-side validation rules not in documentation")
    print("3. Required fields or categories that must exist in your system")


if __name__ == "__main__":
    run_all_tests()