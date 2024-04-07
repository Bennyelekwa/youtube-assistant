from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name ():
    llm = OpenAI(temperature=0.7)

    name = llm("I have a dog as a pet. suggest five cool names for it")

    return name 

if __name__ == "__main__":
    print(generate_pet_name())

# print(2+2)