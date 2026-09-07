
# !pip install langchain langchain-openai biopython faiss-cpu sentence-transformers

import osfrom Bio
import Entrez
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup API Credentials (Replace with your actual keys)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
Entrez.email = "your.email@university.edu"  # Required by NCBI
# 2. Define the Live PubMed Fetcherdef fetch_pubmed_abstracts(query_string, max_results=5):
    """Dynamically queries PubMed and pulls down article abstracts."""
    # Step A: Search for relevant article IDs
    handle = Entrez.esearch(db="pubmed", term=query_string, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]
    
    if not id_list:
        return []
        
    # Step B: Fetch the abstracts for those IDs
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="text")
    raw_data = handle.read()
    handle.close()
    
    # Split raw block text into individual document chunks
    documents = [doc.strip() for doc in raw_data.split("\n\n") if len(doc.strip()) > 50]
    return documents
# 3. Define the LLM Keyword Extractorllm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
keyword_prompt = ChatPromptTemplate.from_template(
    "Extract 2 to 3 medical keywords or mesh terms from these symptoms to search PubMed. "
    "Output ONLY the keywords separated by AND. Do not write anything else.\nSymptoms: {symptoms}"
)keyword_chain = keyword_prompt | llm | StrOutputParser()
# 4. Define the Final RAG Triage Chaintriage_prompt = ChatPromptTemplate.from_template(
    "You are an expert urgent care triage assistant. Review the patient's symptoms alongside "
    "the following dynamic medical literature context.\n\n"
    "Context:\n{context}\n\n"
    "Patient Symptoms:\n{symptoms}\n\n"
    "Assign a triage level (1: Emergency, 2: Urgent, 3: Non-Urgent) and justify it strictly "
    "using the provided context."
)triage_chain = triage_prompt | llm | StrOutputParser()
# 5. Execute the Full JIT Pipelinedef run_triage_pipeline(patient_symptoms):
    print(f"🏥 Processing Symptoms: '{patient_symptoms}'")
    
    # Step 1: Extract keywords via LLM
    search_query = keyword_chain.invoke({"symptoms": patient_symptoms})
    print(f"🔍 Generated PubMed Query: {search_query}")
    
    # Step 2: Fetch raw literature live via API
    raw_docs = fetch_pubmed_abstracts(search_query)
    print(f"📥 Retrieved {len(raw_docs)} articles from PubMed.")
    
    if not raw_docs:
        return "Error: No medical literature found for these symptoms."
        
    # Step 3: Embed data into a temporary in-memory FAISS vector index
    embedding_model = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vector_db = FAISS.from_texts(raw_docs, embedding_model)
    
    # Step 4: Retrieve relevant chunks for the specific symptoms
    retrieved_docs = vector_db.similarity_search(patient_symptoms, k=3)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Step 5: Generate Final Answer
    final_output = triage_chain.invoke({"context": context_text, "symptoms": patient_symptoms})
    return final_output
# --- TEST THE SYSTEM ---symptoms_input = "60 year old female experiencing sudden weakness on the left side of her face and slurred speech."result = run_triage_pipeline(symptoms_input)
print("\n📋 TRIAGE REPORT:\n", result)