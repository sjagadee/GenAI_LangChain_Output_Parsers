from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Initialize the endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    country: str = Field(description="Name of the Country of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name, age and country of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)


# prompt = template.format()

# result = model.invoke(prompt)

# print(result)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | model | parser

result = chain.invoke({})

print(result)