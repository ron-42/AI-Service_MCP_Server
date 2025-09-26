import json
import os
from typing import List, Dict, Any
from datetime import datetime
import hashlib

# Required libraries - install with:
# pip install pinecone-client openai python-dotenv

from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ITServiceDeskKBIngestion:
    def __init__(self):
        """Initialize the ingestion pipeline with API clients."""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = os.getenv('PINECONE_INDEX_NAME')
        
        # Initialize OpenAI
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create or connect to index
        self.index = self._get_or_create_index()
        
        print(f"✅ Connected to Pinecone index: {self.index_name}")
        print(f"✅ OpenAI client initialized")
    
    def _get_or_create_index(self):
        """Get existing index or create a new one if it doesn't exist."""
        try:
            # Try to connect to existing index
            index = self.pc.Index(self.index_name)
            print(f"📋 Found existing index: {self.index_name}")
            return index
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"📝 Index '{self.index_name}' not found. Creating new index...")
                return self._create_index()
            else:
                print(f"❌ Error connecting to index: {e}")
                raise
    
    def _create_index(self):
        """Create a new Pinecone index with appropriate settings."""
        try:
            # OpenAI text-embedding-3-small has 1536 dimensions
            dimension = 1536
            
            # Create index with serverless spec (adjust cloud/region as needed)
            from pinecone import ServerlessSpec
            
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',  # Good for text embeddings
                spec=ServerlessSpec(
                    cloud='aws',      # or 'gcp' depending on your preference
                    region='us-east-1'  # adjust to your preferred region
                )
            )
            
            print(f"✅ Successfully created index '{self.index_name}'")
            print(f"   Dimension: {dimension}")
            print(f"   Metric: cosine")
            print(f"   Spec: Serverless (AWS us-east-1)")
            
            # Wait a moment for index to be ready
            import time
            time.sleep(5)
            
            return self.pc.Index(self.index_name)
            
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            print("💡 You may need to:")
            print("   1. Check your Pinecone plan limits")
            print("   2. Adjust the cloud/region in the spec")
            print("   3. Verify your API key permissions")
            raise
    
    def create_ticket_text_content(self, ticket: Dict[str, Any]) -> str:
        """
        Create a comprehensive text representation of a ticket for embedding.
        """
        # Build the text content
        text_parts = []
        
        # Basic ticket info
        text_parts.append(f"Ticket ID: {ticket['ticketId']}")
        text_parts.append(f"Subject: {ticket['subject']}")
        text_parts.append(f"Category: {ticket['category']}")
        text_parts.append(f"Priority: {ticket['priority']}")
        text_parts.append(f"Status: {ticket['status']}")
        text_parts.append(f"Assigned to: {ticket['assignedTo']}")
        text_parts.append(f"Requester: {ticket['requester']['name']} ({ticket['requester']['email']})")
        text_parts.append(f"Date Reported: {ticket['dateReported']}")
        
        # User description
        text_parts.append(f"User Description: {ticket['userDescription']}")
        
        # Update history
        if ticket.get('updateHistory'):
            text_parts.append("Update History:")
            for update in ticket['updateHistory']:
                text_parts.append(f"- {update}")
        
        # Resolution (if available)
        if ticket.get('resolution'):
            text_parts.append(f"Resolution: {ticket['resolution']}")
        
        # Next steps (if available)
        if ticket.get('nextSteps'):
            text_parts.append(f"Next Steps: {ticket['nextSteps']}")
        
        return "\n".join(text_parts)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI's text-embedding-3-small model.
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise
    
    def create_metadata(self, ticket: Dict[str, Any], dashboard_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for the ticket chunk.
        """
        metadata = {
            # Ticket identifiers
            'ticket_id': ticket['ticketId'],
            'subject': ticket['subject'],
            
            # Categorical information
            'category': ticket['category'],
            'priority': ticket['priority'],
            'status': ticket['status'],
            'assigned_to': ticket['assignedTo'],
            
            # Requester information
            'requester_name': ticket['requester']['name'],
            'requester_email': ticket['requester']['email'],
            
            # Dates
            'date_reported': ticket['dateReported'],
            'ingestion_date': dashboard_info['date'],
            'ingestion_time': dashboard_info['time'],
            
            # Resolution status
            'is_resolved': ticket['status'] == 'Resolved',
            'has_resolution': 'resolution' in ticket and ticket['resolution'] is not None,
            'has_next_steps': 'nextSteps' in ticket and ticket['nextSteps'] is not None,
            
            # Text length for potential filtering
            'description_length': len(ticket['userDescription']),
            'update_count': len(ticket.get('updateHistory', [])),
            
            # Source information
            'source': 'IT_Service_Desk_Dashboard',
            'location': dashboard_info['location']
        }
        
        # Add resolution text if available (truncated for metadata)
        if ticket.get('resolution'):
            metadata['resolution_summary'] = ticket['resolution'][:200]
        
        return metadata
    
    def create_document_id(self, ticket: Dict[str, Any]) -> str:
        """
        Create a unique document ID for the ticket.
        """
        # Use ticket ID as base, add hash of content for uniqueness
        content = self.create_ticket_text_content(ticket)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{ticket['ticketId']}_{content_hash}"
    
    def process_tickets(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process all tickets and prepare them for ingestion.
        """
        dashboard_info = json_data['dashboardInfo']
        tickets = json_data['tickets']
        
        processed_chunks = []
        
        print(f"📊 Processing {len(tickets)} tickets...")
        
        for i, ticket in enumerate(tickets, 1):
            try:
                # Create text content
                text_content = self.create_ticket_text_content(ticket)
                
                # Generate embedding
                embedding = self.generate_embedding(text_content)
                
                # Create metadata
                metadata = self.create_metadata(ticket, dashboard_info)
                
                # Create document ID
                doc_id = self.create_document_id(ticket)
                
                chunk = {
                    'id': doc_id,
                    'values': embedding,
                    'metadata': metadata
                }
                
                processed_chunks.append(chunk)
                print(f"✅ Processed ticket {i}/{len(tickets)}: {ticket['ticketId']}")
                
            except Exception as e:
                print(f"❌ Error processing ticket {ticket['ticketId']}: {e}")
                continue
        
        print(f"🎯 Successfully processed {len(processed_chunks)} tickets")
        return processed_chunks
    
    def upsert_to_pinecone(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upsert processed chunks to Pinecone in batches.
        """
        print(f"🚀 Starting upsert to Pinecone index '{self.index_name}'...")
        
        total_chunks = len(chunks)
        successful_upserts = 0
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                # Upsert batch
                upsert_response = self.index.upsert(vectors=batch)
                successful_upserts += len(batch)
                
                print(f"✅ Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
                print(f"   Response: {upsert_response}")
                
            except Exception as e:
                print(f"❌ Error upserting batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"🎉 Ingestion complete! Successfully upserted {successful_upserts}/{total_chunks} chunks")
        
        # Get index stats
        try:
            stats = self.index.describe_index_stats()
            print(f"📈 Index stats: {stats}")
        except Exception as e:
            print(f"⚠️ Could not retrieve index stats: {e}")
    
    def run_ingestion(self, json_file_path: str):
        """
        Run the complete ingestion pipeline.
        """
        print("🏗️ Starting IT Service Desk KB Ingestion Pipeline")
        print("=" * 60)
        
        # Load JSON data
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            print(f"✅ Loaded JSON data from: {json_file_path}")
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return
        
        # Process tickets
        try:
            chunks = self.process_tickets(json_data)
            if not chunks:
                print("❌ No chunks were processed successfully")
                return
        except Exception as e:
            print(f"❌ Error processing tickets: {e}")
            return
        
        # Upsert to Pinecone
        try:
            self.upsert_to_pinecone(chunks)
        except Exception as e:
            print(f"❌ Error during upsert: {e}")
            return
        
        print("🎊 Pipeline completed successfully!")

def main():
    """
    Main function to run the ingestion pipeline.
    """
    # Check environment variables
    required_env_vars = ['PINECONE_API_KEY', 'PINECONE_INDEX_NAME', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set the following in your .env file:")
        for var in missing_vars:
            print(f"  {var}=your_value_here")
        return
    
    # Initialize and run ingestion
    ingestion = ITServiceDeskKBIngestion()
    
    # Run with the JSON file (update path as needed)
    json_file_path = "/home/ronak/Ronak/Q2-25/mcp-server/kb/1.json"  # Update this to your file path
    ingestion.run_ingestion(json_file_path)

if __name__ == "__main__":
    main()
