from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="nvidia/nemotron-nano-9b-v2:free",
    configuration={
        "base_url": "https://openrouter.ai/api/v1",
    }
)

# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# first propmt - detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# second prompt - summary
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text report. /n {text}",
    input_variables=["text"]
)

prompt1 = template1.format(topic="black hole")

response1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text": response1.content})

response2 = model.invoke(prompt2)

print(response2.content)
