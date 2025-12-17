"""
LLM Agent with RAG Tool Integration

This module defines the customer support agent that uses a Language Model
with Retrieval-Augmented Generation (RAG) capabilities.

Students should implement the RAG tool function and complete the agent setup.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class BaseAgent(ABC):
    """
    Abstract base class for LLM agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            config: Configuration dictionary containing LLM settings, prompts, etc.
        """
        self.config = config or {}
        self.is_initialized = False
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent with LLM and tools."""
        # 1) Initialize LLM (OpenAI chat model by default; override via config)
        model = self.config.get("model", "gpt-3.5-turbo")
        temperature = float(self.config.get("temperature", 0.7))
        api_key = self.config.get("api_key", None)

        # 2) Set up persistent Chroma knowledge base and embeddings (provided helper)
        await self._setup_knowledge_base()

        # 3) Create tools (RAG tool calls self._rag_search)
        tools = await self._create_tools()

        # 4-5) Create ReAct agent + executor with conversational memory
        await self._create_agent(tools)

        self.is_initialized = True
    
    @abstractmethod
    async def process_query(self, text: str, **kwargs) -> str:
        """
        Process a text query and return a response.
        
        Args:
            text: Input text from the user
            **kwargs: Additional context or parameters
            
        Returns:
            str: Agent's response
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")

         # Prefer ainvoke (dict output), fallback to arun for older versions
        if hasattr(self.agent_executor, "ainvoke"):
            result = await self.agent_executor.ainvoke({"input": text, **kwargs})
            return result.get("output", str(result))  # Final Answer
        else:
            result = await self.agent_executor.arun(input=text, **kwargs)
            return result
    
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup agent resources.
        - Clear conversational memory.
        - Drop references to LLM, agent, and executor.
        - Optionally persist/flush or clear Chroma client caches if available.
        """
    # Reset LangChain memory buffer
        if hasattr(self, "memory") and self.memory is not None:
            try:
                # ConversationBufferMemory supports .clear() to wipe history
                self.memory.clear()
            except Exception:
                pass

    # Chroma persistence/caching: generally no close(), but you can persist or clear caches
        try:
            # If using a persistent client, data is already on disk; nothing required.
            # Some versions expose shared cache cleanup helpers; guard them if present.
            import chromadb  # noqa: F401
            if hasattr(getattr(self, "chroma_client", None), "close"):
                self.chroma_client.close()
            # Optional: if API exposes a global cache cleaner, call it safely
            from chromadb.api.client import SharedSystemClient  # type: ignore
            if hasattr(SharedSystemClient, "clear_system_cache"):
                SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        # Drop heavy references
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.collection = None
        self.embedding_model = None
        self.is_initialized = False



class CustomerSupportAgent(BaseAgent):
    """
    Customer Support Agent implementation using LangChain ReAct agent.
    
    This agent uses a Language Model with RAG capabilities to answer
    customer support queries by retrieving relevant information from
    a knowledge base.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.knowledge_base = None
        
    async def initialize(self) -> None:
        """
        TODO: Initialize the customer support agent.
        
        Steps:
        1. Initialize the LLM (e.g., OpenAI, Anthropic, local models)
        2. Set up the knowledge base/vector store
        3. Create RAG tool
        4. Create ReAct agent with tools
        5. Set up agent executor
        """
        # TODO: Initialize LLM
        # Example: from langchain_openai import ChatOpenAI
        # self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.llm = ChatGoogleGenerativeAI( # Gemini via Google AI
            model=self.config.get("model", "gemini-1.5-pro"),
            google_api_key=self.config.get("api_key"), # or picked up from GOOGLE_API_KEY env
            temperature=float(self.config.get("temperature", 0.7)),
        )
        
        # TODO: Initialize knowledge base
        await self._setup_knowledge_base()
        
        # TODO: Create tools including RAG tool
        tools = await self._create_tools()
        
        # TODO: Create agent
        await self._create_agent(tools)
        
        self.is_initialized = True
    
    async def _setup_knowledge_base(self) -> None:
        """
        Set up the knowledge base for RAG using ChromaDB.
        
        This method automatically creates embeddings and stores them in ChromaDB.
        Students only need to implement the retrieval logic in _rag_search().
        """
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            import os
            import hashlib
            
            # Initialize ChromaDB (persistent storage)
            db_path = "./data/chroma_db"
            os.makedirs(db_path, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            
            # Collection name
            collection_name = "customer_support_kb"
            
            # Check if collection already exists and has data
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                if self.collection.count() > 0:
                    print(f"Knowledge base already exists with {self.collection.count()} documents")
                    return
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Customer support knowledge base"}
                )
            
            # Load predefined customer support documents
            knowledge_documents = self._get_customer_support_documents()
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Process and store documents
            print(f"Ingesting {len(knowledge_documents)} documents into knowledge base...")
            
            documents = []
            metadatas = []
            ids = []
            
            for i, doc_data in enumerate(knowledge_documents):
                doc_id = f"doc_{i}_{hashlib.md5(doc_data['content'].encode()).hexdigest()[:8]}"
                
                documents.append(doc_data['content'])
                metadatas.append({
                    'category': doc_data['category'],
                    'title': doc_data['title'],
                    'doc_id': doc_id
                })
                ids.append(doc_id)
            
            # Add documents to ChromaDB (it will automatically create embeddings)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully ingested {len(documents)} documents into ChromaDB")
            
        except Exception as e:
            print(f"Error setting up knowledge base: {str(e)}")
            raise
    
    def _get_customer_support_documents(self) -> List[Dict[str, str]]:
        """
        Predefined customer support knowledge base.
        
        This is the definitive knowledge base that students will work with.
        Do not modify these documents - they form the complete knowledge base.
        """
        return [
            # Return Policy
            {
                "title": "Return Policy Overview",
                "category": "returns",
                "content": "We offer a 30-day return policy for all products purchased from our store. Items must be in original condition with all tags and packaging intact. Returns are processed within 5-7 business days of receiving the returned item. Refunds are issued to the original payment method."
            },
            {
                "title": "Return Process Steps",
                "category": "returns", 
                "content": "To initiate a return: 1) Log into your account and go to Order History, 2) Select the order and click 'Return Items', 3) Choose the items to return and reason, 4) Print the prepaid return label, 5) Pack items securely and attach the label, 6) Drop off at any UPS location or schedule pickup."
            },
            {
                "title": "Non-Returnable Items",
                "category": "returns",
                "content": "The following items cannot be returned: personalized or customized products, perishable goods, digital downloads, gift cards, intimate apparel, and items marked as final sale. Health and safety regulations prevent returns of opened cosmetics and personal care items."
            },
            
            # Shipping Information
            {
                "title": "Shipping Methods and Times",
                "category": "shipping",
                "content": "We offer multiple shipping options: Standard shipping (5-7 business days, free on orders over $50), Express shipping (2-3 business days, $12.99), Next-day shipping (1 business day, $24.99). All orders placed before 2 PM EST ship the same day."
            },
            {
                "title": "International Shipping",
                "category": "shipping",
                "content": "We ship internationally to over 50 countries. International shipping takes 7-14 business days via DHL Express. Shipping costs vary by destination and are calculated at checkout. Customers are responsible for customs fees and import duties. Some restrictions apply to certain products and countries."
            },
            {
                "title": "Order Tracking",
                "category": "shipping",
                "content": "Once your order ships, you'll receive a tracking number via email. Track your package using the tracking number on our website or the carrier's website. You can also track orders by logging into your account and viewing Order History. Tracking updates may take 24 hours to appear."
            },
            
            # Customer Support
            {
                "title": "Contact Information",
                "category": "support",
                "content": "Customer support is available 24/7 via multiple channels: Phone: 1-800-HELP-NOW (1-800-435-7669), Email: support@company.com, Live chat on our website (available 6 AM - 12 AM EST), or submit a support ticket through your account dashboard."
            },
            {
                "title": "Response Times",
                "category": "support",
                "content": "Our support team response times: Live chat - immediate during business hours, Phone support - average wait time under 3 minutes, Email support - response within 4 hours during business days, Support tickets - response within 24 hours. Premium customers receive priority support with faster response times."
            },
            
            # Warranty and Technical Support
            {
                "title": "Product Warranty",
                "category": "warranty",
                "content": "All products come with a manufacturer's warranty. Electronics have 1-year warranty covering defects and malfunctions. Apparel and accessories have 90-day warranty against material defects. Warranty claims require proof of purchase and must be initiated within the warranty period."
            },
            {
                "title": "Technical Support",
                "category": "technical",
                "content": "Free technical support is available for all electronic products. Our certified technicians provide assistance with setup, troubleshooting, and software issues. Technical support is available Monday-Friday 8 AM - 8 PM EST via phone or email. We also offer remote assistance for compatible devices."
            },
            
            # Account and Orders
            {
                "title": "Account Management",
                "category": "account",
                "content": "Manage your account online: Update personal information and addresses, view order history and tracking, manage payment methods, set communication preferences, download invoices and receipts. Account changes may take up to 24 hours to reflect across all systems."
            },
            {
                "title": "Order Modifications",
                "category": "orders",
                "content": "Orders can be modified or canceled within 1 hour of placement if not yet processed. Contact customer service immediately to make changes. Once an order is processed and shipped, it cannot be modified. You can return unwanted items following our return policy."
            },
            
            # Payment and Billing
            {
                "title": "Payment Methods",
                "category": "payment",
                "content": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, Google Pay, and Buy Now Pay Later options through Klarna and Afterpay. Gift cards and store credit can also be used for purchases. Payment is processed securely using 256-bit SSL encryption."
            },
            {
                "title": "Billing and Invoices",
                "category": "billing",
                "content": "Billing occurs when your order ships. You'll receive an email confirmation with invoice details. Invoices are available in your account under Order History. For business purchases, we can provide detailed invoices with tax information. Contact our billing department for any payment disputes or questions."
            },
            
            # Product Information
            {
                "title": "Product Availability",
                "category": "products",
                "content": "Product availability is updated in real-time on our website. If an item shows as 'In Stock', it's available for immediate shipping. 'Limited Stock' means fewer than 10 items remaining. 'Pre-order' items will ship on the specified release date. Out of stock items can be added to your wishlist for restock notifications."
            },
            {
                "title": "Size and Fit Guide",
                "category": "products",
                "content": "Each product page includes detailed size charts and fit information. For apparel, we recommend checking measurements against our size guide rather than relying on size labels from other brands. If you're between sizes, we generally recommend sizing up. Our customer service team can provide personalized fit recommendations."
            }
        ]
    


    async def _create_tools(self) -> List[Tool]:
        """
        Create tools for the agent, including the RAG tool that queries ChromaDB.
        Returns:
            List[Tool]: Tools available to the agent.
        """
        tools: List[Tool] = []

        # Async wrapper so the ReAct agent can await the retrieval
        async def rag_async(query: str) -> str:
            return await self._rag_search(query)

        # LangChain Tool requires a sync func; provide a no-op sync stub and the async coroutine
        rag_tool = Tool(
            name="knowledge_search",
            description="Search the customer support knowledge base for relevant information",
            func=lambda q: "",           # sync fallback (unused in async runs)
            coroutine=rag_async,         # actual async execution path
        )
        tools.append(rag_tool)

        # Placeholder: add additional tools here if needed later
        # e.g., tools.append(Tool(name="order_lookup", description="...", func=..., coroutine=...))

        return tools

    
    async def _rag_search(self, query: str) -> str:
        """
        Retrieve top relevant knowledge base entries from ChromaDB and format them.
        """
        if not hasattr(self, 'collection') or self.collection is None:
            return "Knowledge base not available. Please ensure the service is properly initialized."

        try:
            # 1) Query ChromaDB for top-k results, including docs, metadatas, and distances
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=['documents', 'metadatas', 'distances'],
            )  # Uses Chroma's query_texts path for semantic search [web:15][web:18]

            docs = (results.get('documents') or [[]])[0]
            metas = (results.get('metadatas') or [[]])[0]
            dists = (results.get('distances') or [[]])[0]

            # 2) Handle no hits
            if not docs:
                return "No relevant knowledge base entries found."

            # 3) Format results for readability with title, category, and relevance score
            formatted = []
            for doc, meta, dist in zip(docs, metas, dists):
                title = meta.get('title', 'Untitled')
                category = meta.get('category', 'general')
                # Chroma returns cosine distance (lower is better). Convert to a simple relevance score.
                relevance = 1.0 - float(dist)
                # Optional: truncate very long docs for display
                snippet = doc.strip()
                if len(snippet) > 500:
                    snippet = snippet[:500].rstrip() + "..."
                formatted.append(f"**{title}** [{category}] (relevance: {relevance:.3f})\n{snippet}")

            # 4) Join with spacing for the agent prompt
            return "\n\n".join(formatted)
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    
    async def _create_agent(self, tools: List[Tool]) -> None:
        """
        Create the ReAct agent and wrap it with an AgentExecutor using memory.
        """
        prompt_template = """
        You are a helpful customer support agent. Use the available tools to assist customers.

        You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Question: {input}
        Thought: {agent_scratchpad}
        """
        prompt = PromptTemplate.from_template(prompt_template)

        # Build ReAct agent from model, tools, and prompt
        self.agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt,
        )  # ReAct assembly per LangChain agent docs [web:36][web:47]

        # Wrap with AgentExecutor to handle tool calls + memory
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tools,
            verbose=False,
            memory=self.memory,
        )  # Executor pattern to run agents with tools and state [web:37][web:39]

    
    async def process_query(self, text: str, **kwargs) -> str:
        """
        Process user query using the ReAct agent and return the final answer.
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")

        # Prefer the newer async dict-based interface
        if hasattr(self.agent_executor, "ainvoke"):
            result = await self.agent_executor.ainvoke({"input": text, **kwargs})
            # AgentExecutor typically returns {"output": "...", ...}
            return result.get("output", str(result))
        # Fallback for legacy versions
        return await self.agent_executor.arun(input=text, **kwargs)

    
    async def cleanup(self) -> None:
        """
        Cleanup agent resources:
        - Clear conversation memory
        - Release agent/LLM references
        - Leave Chroma persistent data on disk (no explicit close needed)
        """
        # Clear conversation history to avoid leaking context between sessions
        try:
            if hasattr(self, "memory") and self.memory is not None:
                self.memory.clear()  # ConversationBufferMemory supports clear() [web:37][web:39]
        except Exception:
            pass

        # Chroma PersistentClient stores data on disk; typically no close() required
        # Keep defensive guards in case the client exposes a close() in some versions
        try:
            if getattr(self, "chroma_client", None) and hasattr(self.chroma_client, "close"):
                self.chroma_client.close()  # Optional, version-dependent [web:73][web:18]
        except Exception:
            pass

        # Drop heavy references
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.collection = None
        self.embedding_model = None
        self.is_initialized = False
