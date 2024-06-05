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
