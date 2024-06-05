from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate

from wise_topic import llm_classifier, binary_prompt


llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
question = "Consider a very friendly pet with a fluffy tail. You know it's a cat or a dog. Is it a cat?"
prompt_value = (
    ChatPromptTemplate.from_template(binary_prompt)
    .invoke({"question": question})
    .to_messages()
)

out = llm_classifier(llm, [prompt_value], ["0", "1"])

print(out)
