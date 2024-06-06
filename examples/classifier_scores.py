from langchain_openai import ChatOpenAI

from wise_topic import llm_classifier_binary, llm_classifier_multiple


llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
question1 = "Consider a very friendly pet with a fluffy tail. You know it's a cat or a dog. Is it a cat?"
question2 = "Consider a very friendly pet with a waggy tail. You know it's a cat or a dog. Is it a cat?"
for question in [question1, question2]:
    out = llm_classifier_binary(llm, question)
    print(question)
    print(out)


question3 = "Consider a very friendly pet with a long tail. You know it's a cat, a dog, or a dragon. Which is it?"
out = llm_classifier_multiple(llm, question3, ["cat", "dog", "dragon"])
print(question3)
print(out)

print("done!")
