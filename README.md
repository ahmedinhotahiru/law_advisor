
# Law Advisor Chatbot

Law Advisor Chatbot is an intelligent conversational AI system designed to provide accurate and relevant information about Rwandan laws and legal processes. It leverages LangChain, ChromaDB, OpenAI's GPT-3.5, and other advanced tools to create a retrieval-augmented generation (RAG) system. This system allows users to query legal information across various domains, such as banking laws, company laws, immigration laws, and more.

---

## Features

- **Document Retrieval:** Uses a ChromaDB-powered vectorstore to retrieve legal documents and laws based on user queries.
- **Chatbot Interface:** Offers a conversational AI interface to interact with users, providing detailed and accurate legal guidance.
- **Tool-Based Retrieval:** Specialized retriever tools for various legal domains, including general laws, banking laws, company laws, electronic laws, immigration laws, land/property laws, and tax laws.
- **Custom Embeddings:** Uses OpenAI embeddings for document processing and search.
- **Real-Time Internet Search:** Integrates Tavily Search to fetch general information not covered by the legal databases.
- **Multi-Language Support:** Defaults to English but can respond in other languages based on user input.
- **Persistent Storage:** Maintains a ChromaDB for efficient storage and retrieval of document embeddings.
- **Scalable and Modular:** Designed to allow easy addition of new legal domains or functionalities.

---

## Setup Instructions

### Prerequisites

Ensure you have the following installed:

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ahmedinhotahiru/law_advisor.git
   cd law_advisor
   ```

2. **Install Dependencies:**

   Run the following command to install all required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys:**

   - Create a `.env` file in the root directory.
   - Add your OpenAI and Tavily API keys in the following format:

     ```env
     OPENAI_API_KEY=your_openai_api_key
     TAVILY_API_KEY=your_tavily_api_key
     ```

4. **Prepare Document Directory:**

   - Place your legal documents in the `docs` directory. Ensure the folder structure aligns with the categories defined in the script (e.g., `General Laws`, `Bank Laws`, `Company Laws`, etc.).

5. **Run the Application:**

   Start the chatbot server using Chainlit:

   ```bash
   chainlit run chat_law.py -w
   ```

   Open your browser and navigate to the provided localhost URL to interact with the chatbot.

---

## Directory Structure

```plaintext
law_advisor/
├── chat_law.py            # Main application script
├── requirements.txt       # Python dependencies
├── docs/                  # Folder containing legal documents
│   ├── General Laws/
│   ├── Bank Laws/
│   ├── Company Laws/
│   ├── Electronic Laws/
│   ├── Immigration and Emigration/
│   ├── Land and Property/
│   └── Tax Laws/
├── chroma_db/             # Persistent storage for ChromaDB
├── .env                   # Environment variables for API keys
└── README.md              # Documentation
```

---

## Usage

1. **Chat Interface:**
   - Ask questions related to Rwandan laws and legal processes.
   - Example queries:
     - *"What are the laws regarding business registration in Rwanda?"*
     - *"Tell me about electronic transaction laws in Rwanda."*

2. **Tool Behavior:**
   - The chatbot automatically identifies the appropriate retriever tool based on your query.
   - If no specific tool applies, it defaults to using the Tavily Search tool for general information.

3. **Languages:**
   - Responds in English by default.
   - Automatically switches to the language of the query if detected.

---

## How It Works

1. **Document Ingestion:**
   - Legal documents are loaded from the `docs` directory.
   - Text is extracted using PyPDFLoader and split into chunks using RecursiveCharacterTextSplitter.
   - Chunks are embedded using OpenAI embeddings and stored in ChromaDB.

2. **Tool Initialization:**
   - Each legal domain (e.g., banking laws, tax laws) is treated as a separate vectorstore retriever.
   - A unique retriever tool is created for each domain.

3. **Chat Agent:**
   - The chatbot uses a predefined system prompt to ensure it responds accurately and effectively.
   - The system dynamically selects and invokes the appropriate tools based on the user query.

4. **Tavily Integration:**
   - For non-legal queries or topics outside the scope of the documents, the chatbot uses the Tavily Search API to fetch real-time information from the internet.

---

## Future Enhancements

- **Expand Legal Coverage:** Add more domains and laws to the dataset.
- **Advanced NLP Models:** Incorporate fine-tuned legal language models for better context understanding.
- **Multilingual Support:** Enhance support for Kinyarwanda and French to cater to local needs.

---

## Acknowledgments

- **LangChain** for providing powerful libraries for building AI pipelines.
- **ChromaDB** for robust vectorstore management.
- **OpenAI** for GPT-3.5 and embedding models.
- **Chainlit** for creating the interactive chatbot interface.
- **Tavily** for real-time internet search capabilities.

---

## Team

- Ahmed Tahiru Issah (aissah@andrew.cmu.edu)
- Leonard Niyitegeka (lniyiteg@andrew.cmu.edu)
- Ngoga Alexis (nalexis@andrew.cmu.edu)
- Mohammed Hardi Abdul Baaki (mabdulba@andrew.cmu.edu)

## Contact

For queries or support, reach out to any member of the team via the emails listed above
