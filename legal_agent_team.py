import streamlit as st
from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.qdrant import Qdrant
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
import tempfile
import os

#initializing the session state variables
def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = None
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = None
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None

def init_qdrant():
    """Initialize Qdrant vector database"""
    if not st.session_state.qdrant_api_key:
        raise ValueError("Qdrant API key not provided")
    if not st.session_state.qdrant_url:
        raise ValueError("Qdrant URL not provided")
        
    try:
        # First try direct connection with QdrantClient
        st.write("Testing direct connection first...")
        from qdrant_client import QdrantClient
        
        direct_client = QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=20
        )
        
        # Test direct connection
        collections = direct_client.get_collections()
        st.write(f"Direct connection successful! Found collections: {[c.name for c in collections.collections]}")
        
        # Now try with Phi's Qdrant wrapper
        st.write("\nInitializing Phi Qdrant wrapper...")
        vector_db = Qdrant(          
            collection="legal_knowledge",
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            https=True,
            timeout=20,
            distance="cosine"
        )
        
        # Test connection and create collection if needed
        st.write("Testing Phi Qdrant wrapper connection...")
        collections = vector_db.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        st.write(f"Found collections: {collection_names}")
        
        if "legal_knowledge" not in collection_names:
            st.write("Creating 'legal_knowledge' collection...")
            vector_db.client.create_collection(
                collection_name="legal_knowledge",
                vectors_config={
                    "size": 1536,  # OpenAI embedding dimension
                    "distance": "Cosine"
                }
            )
            st.write("Collection created successfully!")
            
        return vector_db
        
    except Exception as e:
        import traceback
        st.error("=== Detailed Error Information ===")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Error message: {str(e)}")
        if hasattr(e, 'response'):
            st.error(f"Response content: {e.response.content}")
        st.error(f"Stack trace:\n{traceback.format_exc()}")
        raise Exception(f"Qdrant connection failed: {str(e)}")

def process_document(uploaded_file, vector_db: Qdrant):
    """Process document, create embeddings and store in Qdrant vector database"""
    if not st.session_state.openai_api_key:
        raise ValueError("OpenAI API key not provided")
        
    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    
    # Test Qdrant connection first
    try:
        st.write("Testing Qdrant connection...")
        collections = vector_db.client.get_collections()
        st.write(f"Available collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        raise Exception(f"Qdrant connection test failed: {str(e)}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        st.write(f"Created temporary directory: {temp_dir}")
        
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"Saved uploaded file to: {temp_file_path}")

        try:
            st.write("Initializing OpenAI embedder...")
            embedder = OpenAIEmbedder(
                model="text-embedding-3-small",
                api_key=st.session_state.openai_api_key
            )
            
            st.write("Creating knowledge base...")
            # Creating knowledge base with explicit Qdrant configuration
            knowledge_base = PDFKnowledgeBase(
                path=temp_dir, 
                vector_db=vector_db, 
                reader=PDFReader(chunk=True),
                embedder=embedder,
                recreate_vector_db=True  
            )
            
            st.write("Loading knowledge base...")
            knowledge_base.load()     
            st.write("Knowledge base loaded successfully!")
            return knowledge_base      
        except Exception as e:
            st.error(f"Detailed error: {str(e)}")
            import traceback
            st.error(f"Stack trace: {traceback.format_exc()}")
            raise Exception(f"Error processing document: {str(e)}")

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 API Configuration")
   
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="Enter your OpenAI API key"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key

        qdrant_key = st.text_input(
            "Qdrant API Key",
            type="password",
            value=st.session_state.qdrant_api_key if st.session_state.qdrant_api_key else "",
            help="Enter your Qdrant API key"
        )
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key

        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url if st.session_state.qdrant_url else "https://34539fef-443a-4399-a418-9a5a44bdf3c6.us-west-1-0.aws.cloud.qdrant.io:6333",
            help="Enter your Qdrant instance URL"
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url

        if all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
            try:
                if not st.session_state.vector_db:
                    st.session_state.vector_db = init_qdrant()
                    st.success("Successfully connected to Qdrant!")
            except Exception as e:
                st.error(f"Failed to connect to Qdrant: {str(e)}")

        st.divider()

        if all([st.session_state.openai_api_key, st.session_state.vector_db]):
            st.header("📄 Document Upload")
            uploaded_file = st.file_uploader("Upload Legal Document", type=['pdf'])
            
            if uploaded_file:
                with st.spinner("Processing document..."):
                    try:
                        knowledge_base = process_document(uploaded_file, st.session_state.vector_db)
                        st.session_state.knowledge_base = knowledge_base
                        
                        # Initialize agents
                        legal_researcher = Agent(
                            name="Legal Researcher",
                            role="Legal research specialist",
                            model=OpenAIChat(model="gpt-4o"),
                            tools=[DuckDuckGo()],
                            knowledge=st.session_state.knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Find and cite relevant legal cases and precedents",
                                "Provide detailed research summaries with sources",
                                "Reference specific sections from the uploaded document",
                                "Always search the knowledge base for relevant information"
                            ],
                            show_tool_calls=True,
                            markdown=True
                        )

                        contract_analyst = Agent(
                            name="Contract Analyst",
                            role="Contract analysis specialist",
                            model=OpenAIChat(model="gpt-4o"),
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Review contracts thoroughly",
                                "Identify key terms and potential issues",
                                "Reference specific clauses from the document"
                            ],
                            markdown=True
                        )

                        legal_strategist = Agent(
                            name="Legal Strategist", 
                            role="Legal strategy specialist",
                            model=OpenAIChat(model="gpt-4o"),
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Develop comprehensive legal strategies",
                                "Provide actionable recommendations",
                                "Consider both risks and opportunities"
                            ],
                            markdown=True
                        )

                        # Legal Agent Team
                        st.session_state.legal_team = Agent(
                            name="Legal Team Lead",
                            role="Legal team coordinator",
                            model=OpenAIChat(model="gpt-4o"),
                            team=[legal_researcher, contract_analyst, legal_strategist],
                            knowledge=st.session_state.knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Coordinate analysis between team members",
                                "Provide comprehensive responses",
                                "Ensure all recommendations are properly sourced",
                                "Reference specific parts of the uploaded document",
                                "Always search the knowledge base before delegating tasks"
                            ],
                            show_tool_calls=True,
                            markdown=True
                        )
                        
                        st.success("✅ Document processed and team initialized!")
                            
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")

            st.divider()
            st.header("🔍 Analysis Options")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Contract Review",
                    "Legal Research",
                    "Risk Assessment",
                    "Compliance Check",
                    "Custom Query"
                ]
            )
        else:
            st.warning("Please configure all API credentials to proceed")

    # Main content area
    if not all([st.session_state.openai_api_key, st.session_state.vector_db]):
        st.info("👈 Please configure your API credentials in the sidebar to begin")
    elif not uploaded_file:
        st.info("👈 Please upload a legal document to begin analysis")
    elif st.session_state.legal_team:
        # Create a dictionary for analysis type icons
        analysis_icons = {
            "Contract Review": "📑",
            "Legal Research": "🔍",
            "Risk Assessment": "⚠️",
            "Compliance Check": "✅",
            "Custom Query": "💭"
        }

        # Dynamic header with icon
        st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
  
        analysis_configs = {
            "Contract Review": {
                "query": "Review this contract and identify key terms, obligations, and potential issues.",
                "agents": ["Contract Analyst"],
                "description": "Detailed contract analysis focusing on terms and obligations"
            },
            "Legal Research": {
                "query": "Research relevant cases and precedents related to this document.",
                "agents": ["Legal Researcher"],
                "description": "Research on relevant legal cases and precedents"
            },
            "Risk Assessment": {
                "query": "Analyze potential legal risks and liabilities in this document.",
                "agents": ["Contract Analyst", "Legal Strategist"],
                "description": "Combined risk analysis and strategic assessment"
            },
            "Compliance Check": {
                "query": "Check this document for regulatory compliance issues.",
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Comprehensive compliance analysis"
            },
            "Custom Query": {
                "query": None,
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Custom analysis using all available agents"
            }
        }

        st.info(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 Active Legal AI Agents: {', '.join(analysis_configs[analysis_type]['agents'])}")  #dictionary!!

        # Replace the existing user_query section with this:
        if analysis_type == "Custom Query":
            user_query = st.text_area(
                "Enter your specific query:",
                help="Add any specific questions or points you want to analyze"
            )
        else:
            user_query = None  # Set to None for non-custom queries


        if st.button("Analyze"):
            if analysis_type == "Custom Query" and not user_query:
                st.warning("Please enter a query")
            else:
                with st.spinner("Analyzing document..."):
                    try:
                        # Ensure OpenAI API key is set
                        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                        
                        # Combine predefined and user queries
                        if analysis_type != "Custom Query":
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            Primary Analysis Task: {analysis_configs[analysis_type]['query']}
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            Please search the knowledge base and provide specific references from the document.
                            """
                        else:
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            {user_query}
                            
                            Please search the knowledge base and provide specific references from the document.
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """

                        response = st.session_state.legal_team.run(combined_query)
                        
                        # Display results in tabs
                        tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                        
                        with tabs[0]:
                            st.markdown("### Detailed Analysis")
                            if response.content:
                                st.markdown(response.content)
                            else:
                                for message in response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[1]:
                            st.markdown("### Key Points")
                            key_points_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:    
                                {response.content}
                                
                                Please summarize the key points in bullet points.
                                Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if key_points_response.content:
                                st.markdown(key_points_response.content)
                            else:
                                for message in key_points_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[2]:
                            st.markdown("### Recommendations")
                            recommendations_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:
                                {response.content}
                                
                                What are your key recommendations based on the analysis, the best course of action?
                                Provide specific recommendations from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if recommendations_response.content:
                                st.markdown(recommendations_response.content)
                            else:
                                for message in recommendations_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    else:
        st.info("Please upload a legal document to begin analysis")

    # Add debug section
    st.divider()
    st.header("🔧 Debug Info")
    if st.checkbox("Show Connection Details"):
        st.code(f"""
URL: {st.session_state.qdrant_url}
API Key: {st.session_state.qdrant_api_key[:8]}... (truncated)
Vector DB Status: {"Initialized" if st.session_state.vector_db else "Not Initialized"}
        """)
        
    if st.button("Clear Credentials"):
        st.session_state.qdrant_api_key = None
        st.session_state.qdrant_url = None
        st.session_state.vector_db = None
        st.session_state.openai_api_key = None
        st.experimental_rerun()

if __name__ == "__main__":
    main() 

# EXWLgeyMKAkkdMoiIyNRipxhE0iiqWnx032linLWDckYqrAKiwwh8Q
# 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