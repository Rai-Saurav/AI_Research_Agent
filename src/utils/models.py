from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatMistralAI(model='ministral-8b-2512',temperature=0)