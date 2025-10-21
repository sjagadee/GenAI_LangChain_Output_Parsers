from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Initialize the endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

# Wrap it in ChatHuggingFace for chat functionality
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is machine learning?")
print(response.content)