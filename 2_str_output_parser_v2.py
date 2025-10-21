from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

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

parser = StrOutputParser()

# this is chain - which creates a pipeline and passes the previous output to the next in lu=
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'blackhole'})

print(result)
