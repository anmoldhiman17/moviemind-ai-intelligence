from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(model='mistral-small-2506')

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful structured information from a movie paragraph and present it in a clean readable format.

Rules:
- Do NOT add explanations
- Do NOT add extra commentary
- Follow the exact format
- If information is missing → write NULL
- Keep summary short (2–3 lines max)
- Do NOT guess unknown facts

Output Format:

Movie Title:
Release Year:
Genre:
Director:
Main Cast:
Setting/Location:
Plot:
Themes:
Ratings:
Notable Features:

Short Summary:
"""
    ),   

    ("human",
     """
Extract information from this paragraph:

{paragraph}
"""
    )
])

para = input("Give your paragraph : ")

final_prompt = prompt.invoke({"paragraph": para})

response = model.invoke(final_prompt)

print(response.content)