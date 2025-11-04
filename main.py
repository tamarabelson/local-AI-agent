from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver

model = OllamaLLM(model = "llama3.2")

template = """
You are an expert  in answering questions about a piza restaurant

Here are some relevant reviews: {reviews}

Here is some question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    review = retriver.invoke(question)
    result = chain.invoke({"reviews": [], "question": question})
    print(result)
