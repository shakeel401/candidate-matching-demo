from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY"))

# Prompt to extract only relevant info
prompt_template = PromptTemplate.from_template("""
You are a smart resume processor. Extract only the most relevant and important information from this resume that would help match it to a job description. Include key skills, job titles, education, certifications, and work experience. 
Also, extract and include the LinkedIn profile URL if available. Return clean, readable text.

Resume Text:
{resume_text}
""")

# Modern way to chain
extract_chain = prompt_template | llm

def load_resume(file_path: str, user_id: str = None) -> list[Document]:
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file format: Only .pdf and .docx are allowed")

    raw_docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in raw_docs])

    # Extract relevant resume content
    result = extract_chain.invoke({"resume_text": full_text})
    relevant_text = result.content  # updated output format from ChatOpenAI

    resume_id = str(uuid.uuid4())
    file_name = os.path.basename(file_path)

    document = Document(
        page_content=relevant_text,
        metadata={
            "filename": file_name,
            "resume_id": resume_id,
            "user_id": user_id or "unknown"
        }
    )

    return [document]
