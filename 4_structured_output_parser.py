from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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

schema = [
    ResponseSchema(name="fact_1", description="The first fact about the topic"),
    ResponseSchema(name="fact_2", description="The second fact about the topic"),
    ResponseSchema(name="fact_3", description="The third fact about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)


# prompt = template.format({'topic': 'black hole'})

# result = model.invoke(prompt)

# print(result)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)