import streamlit as st
import os
from dotenv import load_dotenv
from utils.parse_resume import load_resume
from utils.pinecone_ops import insert_documents, search_similar_docs
from utils.query_generator import generate_query_from_jd
from openai import OpenAI
import json
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredURLLoader

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page config
st.set_page_config(page_title="🧠 AI Candidate Matcher", layout="centered")
st.title("🧠 AI Candidate Matcher")
st.markdown("Smartly match job descriptions with top candidates using GPT-4o, Pinecone & LangChain.")

# Sidebar for action
st.sidebar.title("⚙️ Actions")
option = st.sidebar.radio("Choose mode", ["Upload Resumes", "Match Job Description"])

# === Resume Upload Mode ===
if option == "Upload Resumes":
    st.header("📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process and Upload") and uploaded_files:
        st.info("Processing resumes...")
        for file in uploaded_files:
            os.makedirs("data/resumes", exist_ok=True)
            path = os.path.join("data/resumes", file.name)
            with open(path, "wb") as f:
                f.write(file.read())

            try:
                docs = load_resume(path)
                insert_documents(docs)
                st.success(f"✅ Uploaded: {file.name}")
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")

# === Job Matching Mode ===
elif option == "Match Job Description":
    st.header("Match Job Description to Candidates")
    st.markdown("Provide job description by any **one** method below:")

    col1, col2, col3 = st.columns(3)
    with col1:
        jd_text = st.text_area("📝 Paste Job Description", height=230)
    with col2:
        jd_file = st.file_uploader("📎 Upload JD File", type=["pdf", "docx"])
    with col3:
        jd_url = st.text_input("🌐 JD URL (LinkedIn, etc.)")

    job_description = ""

    if st.button("🔍 Find Best Candidates"):
        try:
            # Load job description
            if jd_text.strip():
                job_description = jd_text

            elif jd_file:
                os.makedirs("data/jobdesc", exist_ok=True)
                path = os.path.join("data/jobdesc", jd_file.name)
                with open(path, "wb") as f:
                    f.write(jd_file.read())
                loader = UnstructuredFileLoader(path)
                docs = loader.load()
                if not docs or not docs[0].page_content.strip():
                    st.error("❌ Failed to extract content from the uploaded file.")
                    st.stop()
                job_description = docs[0].page_content

            elif jd_url.strip():
                loader = UnstructuredURLLoader(urls=[jd_url])
                docs = loader.load()
                if not docs or not docs[0].page_content.strip():
                    st.error("❌ Failed to fetch or extract content from the provided URL.")
                    st.stop()
                job_description = docs[0].page_content

            else:
                st.warning("⚠️ Please provide a job description.")
                st.stop()

            # Query generation and search
            with st.spinner("🔎 Analyzing and matching candidates..."):
                query = generate_query_from_jd(job_description)
                results = search_similar_docs(query)

        except Exception as e:
            st.error(f"❌ Error during job matching: {e}")
            results = []

        # Results
        if results:
            st.subheader("🏆 Top Matching Candidates")
            for i, res in enumerate(results):
                confidence = res.get("confidence_score", 0)
                content = res.get("resume_text", "")
                metadata = res.get("metadata", {})

                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Extract and return raw JSON (no markdown, no explanations) "
                                    "with these fields: name, title, linkedin, email, phone."
                                )
                            },
                            {"role": "user", "content": content}
                        ],
                        temperature=0.2
                    )
                    raw_response = completion.choices[0].message.content.strip()
                    # Safe parsing
                    try:
                        structured = json.loads(raw_response)
                    except json.JSONDecodeError:
                        structured = {"error": "Invalid JSON"}
                        st.warning("⚠️ Model returned invalid JSON.")
                        st.code(raw_response)

                except Exception as e:
                    structured = {"error": f"❌ Parsing error: {str(e)}"}

                # Display candidate card
                with st.container():
                    st.markdown("------")
                    if "error" in structured:
                        st.error(structured["error"])
                    else:
                        st.markdown(f"""
                        **👤 Name:** `{structured.get("name", "N/A")}`  
                        **🎯 Title:** `{structured.get("title", "N/A")}`  
                        **📧 Email:** `{structured.get("email", "N/A")}`  
                        **📞 Phone:** `{structured.get("phone", "N/A")}`  
                        **🔗 LinkedIn:** {structured.get("linkedin", "N/A")}  
                        **📊 Match Confidence:** `{confidence:.2f}%`
                        """)

        else:
            st.info("🤷‍♂️ No matching candidates found. Try refining the job description.")
