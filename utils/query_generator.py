from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_query_from_jd(job_description):
    prompt = f"""
    Extract the core skills, tools, and experience required from the following job description.
    Return them as a single string that could be used to search for matching candidate resumes.
    ---
    {job_description}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()