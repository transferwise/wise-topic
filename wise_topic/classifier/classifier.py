from typing import List

import numpy as np

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel


def llm_classifier(llm: BaseChatModel, messages, valid_options, top_logprobs=5, max_tokens=1):
    result = llm.generate(
        messages,
        logprobs=True,
        top_logprobs=top_logprobs,
        max_tokens=max_tokens,
    )
    info = result.generations[0][0].generation_info["logprobs"]["content"][0][
        "top_logprobs"
    ]

    scores = logprobs_to_scores(info, valid_options)
    return scores


def logprobs_to_scores(logprobs, valid_options: List[str]):
    scores = np.array(len(valid_options) * [float("-inf")])
    matches = False
    for i, c in enumerate(valid_options):
        for p in logprobs:
            if isinstance(p, dict): # Langchain interface
                token = p["token"]
                logprob = p["logprob"]
            else:  # OpenAI interface
                token = p.token
                logprob = p.logprob
            if token == c:
                matches = True
                scores[i] = logprob
    if matches:
        scores = scores - np.max(scores)
        scores = np.exp(scores)
        scores = scores / np.sum(scores)
    else:  # If no matches, return uniform distribution - is that optimal?
        scores = np.ones(len(valid_options)) / len(valid_options)

    return scores

    # And this is how to do this with openai direct
    # response = raw_llm.chat.completions.create(
    #     model="gpt-4-1106-preview",
    #     messages=[
    #         {"role": "user", "content": reasoning_prompt.to_string()},
    #     ],
    #     logprobs=True,
    #     top_logprobs=5,
    #     max_tokens=1,
    # )
    # info = response.choices[0].logprobs.content[0].top_logprobs


def classifier_with_reasoning(
    llm, reasoning_prompt: str, binary_prompt: str, reasoning_args: dict
):

    reasoning_template = PromptTemplate.from_template(reasoning_prompt)
    reasoning_prompt = reasoning_template.invoke(reasoning_args).to_messages()
    chain = reasoning_template | llm
    reasoning = chain.invoke(reasoning_args)
    messages = [reasoning_prompt + [reasoning, HumanMessage(content=binary_prompt)]]
    scores = llm_classifier(llm, messages, ["0", "1"])
    out = {
        "reasoning": reasoning.content,
        "prob(1)": scores[1],
    }

    return out
