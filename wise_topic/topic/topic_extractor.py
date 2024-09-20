import json
from json import JSONDecodeError
from typing import Sequence, Optional
from math import ceil

from langchain_core.language_models import BaseChatModel
from langchain.prompts import PromptTemplate


def extract_topics(
    docs: Sequence[str],
    llm: BaseChatModel,
    extra_prompt: str | None = None,
    n_topics=3,
    max_words=5,
    with_count: bool = False,
):
    min_topics = ceil(n_topics / 2)
    docs_ = ";\n".join(docs)
    prompt_text = f"""I am about to give you a list of documents, separated by semicolons
    and line breaks, and jointly delimited by triple back quotes.

          After the final document, there will be the word <END>

          Generate a list of at most {n_topics} and at least {min_topics} distinct, non-overlapping topics
        that between them best describe the content of the documents.
        {extra_prompt if extra_prompt is not None else ""}
        Each topic should have at least 3 and at most {max_words} words.
        Avoid composite topics (`something and something else`)
        """
    if with_count:
        prompt_text += """

        Each topic should be followed by a count of how many documents it applies to

        Output the topics in the following format:
        ```
        {{"topic1":count_of_documents_with_topic1, ...}}
        ```
        """
    else:
        prompt_text += """
        Output the topics in the following format:
        ```
        ["topic1", "topic2", ...]
        ```
        """
    prompt_text += """Don't exceed the {max_words} words limit, including words such as 'and' and 'of'.
        Make sure each following topic does not overlap with or duplicate any of the previous topics.
       After seeing <END>, output a well-formed json containing the list of topics as described above.
       Make sure to return between {min_topics} and {n_topics} distinct, non-overlapping topics.
       ```{docs}
        ```
        <END>"""
    if hasattr(llm, "max_tokens"):
        llm.max_tokens = 5 * (max_words + 1) * n_topics

    chain = PromptTemplate.from_template(prompt_text) | llm
    for _ in range(3):
        try:
            out = chain.invoke(
                {
                    "docs": docs_,
                    "n_topics": n_topics,
                    "max_words": max_words,
                    "min_topics": min_topics,
                }
            )
            return json.loads(out.content.replace("```json", "").replace("```", ""))
        except JSONDecodeError:
            pass
