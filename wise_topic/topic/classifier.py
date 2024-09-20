import logging
from typing import Sequence

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel


def topic_classifier(topics: Sequence[str], message: str, llm: BaseChatModel):
    re_topics = "\n".join([f"{i+1}. {t}" for i, t in enumerate(topics)])
    prompt_text = """I am about to give you a numbered list of topics.
        Then I will pass to you a message, after the word MESSAGE.
        Return an integer that is the number of the topic that best fits that message;
        if no topic fits the message, return 0.

        Return only the number, without additional text.
        {categories}
        MESSAGE:
        {message}
        TOPIC NUMBER:"""

    if hasattr(llm, "max_tokens"):
        llm.max_tokens = 5

    chain = PromptTemplate.from_template(prompt_text) | llm
    out = chain.invoke({"categories": re_topics, "message": message})
    try:
        return int(out.content)
    except Exception as e:
        logging.warning(f"Error in classifier: {e}")
        return 0


async def topic_classifier_async(
    topics: Sequence[str], message: str, llm: BaseChatModel
) -> int:
    """
    Asynchronously classify the message into one of the provided topics using a language model.
    Return an integer corresponding to the number of the best fitting topic.
    If no topic fits, return 0.

    Parameters:
    - topics (Sequence[str]): List of possible topics.
    - message (str): The message to classify.
    - llm (BaseChatModel): The language model used for classification.

    Returns:
    - int: The number corresponding to the best-fitting topic, or 0 if no fit.
    """
    re_topics = "\n".join([f"{i+1}. {t}" for i, t in enumerate(topics)])
    prompt_text = """I am about to give you a numbered list of topics.
        Then I will pass to you a message, after the word MESSAGE.
        Return an integer that is the number of the topic that best fits that message;
        if no topic fits the message, return 0.

        Return only the number, without additional text.
        {categories}
        MESSAGE:
        {message}
        TOPIC NUMBER:"""

    if hasattr(llm, "max_tokens"):
        llm.max_tokens = 5

    # Create the prompt chain using the template and language model
    chain = PromptTemplate.from_template(prompt_text) | llm

    try:
        # Invoke the language model asynchronously
        out = await chain.ainvoke({"categories": re_topics, "message": message})
        return int(out.content)
    except Exception as e:
        logging.warning(f"Error in classifier: {e}")
        return 0


# Run the example
if __name__ == "__main__":
    import asyncio
    from langchain_openai import ChatOpenAI

    async def main(llm):

        # Simulated topics and message
        topics = ["Sports", "Technology", "Music", "Politics"]
        message = "Artificial intelligence is transforming industries."
        # Call the topic_classifier with the mock LLM
        result = await topic_classifier_async(topics, message, llm)

        # Return the resulting topic number
        return result

    classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = asyncio.run(main(classifier_llm))
    print(f"Topic number: {result}")
