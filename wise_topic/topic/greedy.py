import asyncio
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Sequence, Callable, Union
import logging
import copy

import numpy as np
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from .classifier import topic_classifier, topic_classifier_async
from .topic_extractor import extract_topics
from ..parallel import process_batch_parallel


class GreedyTopicExtractor:
    def __init__(
        self,
        max_unclassified_messages: int = 30,
        initial_unclassified_messages: Optional[int] = None,
        num_topics_per_update: int = 3,
        initial_num_topics: Optional[int] = None,
        max_words_in_topic: int = 6,
        initial_topics: Optional[Sequence[str]] = None,
        topic_extractor: Callable = extract_topics,
        topic_llm: Union[str, BaseLanguageModel] = "gpt-4o",
        classifier: Callable = topic_classifier,
        classifier_llm: Union[str, BaseLanguageModel] = "gpt-4o-mini",
        max_parallel_calls: int = 5,
        extra_prompt: str | None = None,
        verbose: bool = False,
    ):
        """

        :param max_unclassified_messages:
        :param initial_unclassified_messages:
        :param num_topics_per_update:
        :param initial_num_topics:
        :param max_words_in_topic:
        :param initial_topics:
        :param topic_extractor:
        :param topic_llm:
        :param classifier:
        :param classifier_llm:
        :param verbose:
        """
        self.max_unclassified_messages = max_unclassified_messages
        self.initial_unclassified_messages = (
            self.max_unclassified_messages * 2
            if initial_unclassified_messages is None
            else initial_unclassified_messages
        )
        self.num_topics_per_update = num_topics_per_update
        self.initial_num_topics = (
            num_topics_per_update * 2
            if initial_num_topics is None
            else initial_num_topics
        )
        self.max_words_in_topic = max_words_in_topic
        self.topic_extractor = topic_extractor
        if isinstance(topic_llm, str):
            topic_llm = ChatOpenAI(model=topic_llm, temperature=0)
        self.topic_llm = topic_llm
        self.classifier = classifier
        if isinstance(classifier_llm, str):
            classifier_llm = ChatOpenAI(model=classifier_llm, temperature=0)
        self.classifier_llm = classifier_llm
        self.extra_prompt = extra_prompt
        self.verbose = verbose
        self.max_parallel_calls = max_parallel_calls

        self.topics = [] if initial_topics is None else list(initial_topics)
        self.messages_to_topics = {}
        self.unclassified_messages = []
        self.rename = True

    def __call__(self, messages: Sequence[str]):
        """
        Processes a list of messages synchronously while allowing up to N simultaneous async classifier calls.
        Pauses all classify_message calls if topics are being updated or extracted. Handles both cases where
        an event loop is already running or needs to be created.

        Parameters:
        - messages (Sequence[str]): The list of messages to classify.
        - max_concurrent_tasks (int): Maximum number of simultaneous async classifier calls.
        """
        np.random.shuffle(messages)

        # Process messages in batches, assigning them to existing topics
        # if too many messages that don't fit have accumulated, extract more topics from them
        # and add them to the list
        for ind in range(0, len(messages), self.max_unclassified_messages):
            msg = messages[
                ind : min(ind + self.max_unclassified_messages, len(messages))
            ]
            self.process_batch(msg, self.max_parallel_calls)

            # Check if topics need to be updated
            update_topics = False
            if len(self.topics):
                if len(self.unclassified_messages) > self.max_unclassified_messages:
                    update_topics = True
            else:
                if len(self.unclassified_messages) > self.initial_unclassified_messages:
                    update_topics = True
                elif len(messages) - ind < self.max_unclassified_messages:

                    # we're done and there were not sufficient messages in total to trigger the above
                    update_topics = True

            # Pause tasks and update topics synchronously if necessary
            if update_topics:
                logging.info("Updating topics...")
                new_topics = self.extract_topics(
                    self.unclassified_messages, extra_prompt=self.extra_prompt
                )
                self.topics += new_topics
                print("new topics:", new_topics)
                unclassified = self.unclassified_messages
                self.unclassified_messages = []
                self.process_batch(unclassified, self.max_parallel_calls)
                logging.info(self.topic_counts)

        # after we've finished classifying, rename the topics to reflect all the messages under them
        t2m = self.topics_to_messages
        for i, t in enumerate(copy.copy(self.topics)):
            if t != "Other":
                new_name = self.extract_topics(t2m[t])[0]
                self.topics[i] = new_name

        # Log topic counts after processing all messages
        logging.info(self.topic_counts)

    def process_batch(self, msg, n_jobs: int = 5):
        def classify_message(message: str):
            if len(self.topics):
                topic_num = topic_classifier(
                    self.topics, message, llm=self.classifier_llm
                )
                assert topic_num <= len(self.topics)
            else:
                topic_num = 0

            return topic_num

        results = process_batch_parallel(
            classify_message, [(m,) for m in msg], max_workers=n_jobs
        )

        for topic_num, (message,) in results:
            if topic_num == 0:
                self.unclassified_messages.append(message)
            else:
                self.messages_to_topics[message] = topic_num - 1
                if self.verbose:
                    print(message, "\n", self.topics[topic_num - 1], "\n***********")

    def update_topics(self):
        messages = self.unclassified_messages
        self.unclassified_messages = []

        new_topics = self.extract_topics(messages)
        self.topics += new_topics

        for m in messages:
            self.step(m)

    def extract_topics(self, messages: Sequence[str], extra_prompt: str | None = None):
        logging.info("updating topics...")

        num_topics = (
            self.num_topics_per_update if len(self.topics) else self.initial_num_topics
        )

        topics = self.topic_extractor(
            messages,
            extra_prompt=extra_prompt,
            n_topics=num_topics,
            with_count=True,
            max_words=self.max_words_in_topic,
            llm=self.topic_llm,
        )
        if self.verbose:
            print(topics)
        return list(topics.keys())

    @property
    def topics_to_messages(self):
        out = defaultdict(list)
        for m, t in self.messages_to_topics.items():
            out[self.topics[t]].append(m)

        if len(self.unclassified_messages):
            out["Other"] = self.unclassified_messages
        return out

    @property
    def topic_counts(self):
        return {k: len(v) for k, v in self.topics_to_messages.items()}


def greedy_topic_tree(messages, max_depth=0, **kwargs):

    initial_topics = kwargs.pop("initial_topics", None)
    assert initial_topics is None or isinstance(initial_topics, (list, tuple, dict))

    for key in ["max_unclassified_messages", "initial_unclassified_messages"]:
        if key in kwargs and kwargs[key] < 1:
            kwargs[key] = int(len(messages) * kwargs[key])

    gte = GreedyTopicExtractor(
        **kwargs,
        initial_topics=(
            list(initial_topics.keys())
            if isinstance(initial_topics, dict)
            else initial_topics
        ),
    )
    gte(messages)
    topic_tree = {}
    t2m = gte.topics_to_messages
    for t, m in t2m.items():
        topic_tree[t] = {"messages": m}

    if max_depth > 1 and len(topic_tree) > 1:  # to prevent endless recursion
        for t, m in t2m.items():
            if len(m) > gte.max_unclassified_messages:
                logging.info(f"Expanding topic {t} with {len(m)} messages")
                candidate_tree = greedy_topic_tree(m, max_depth=max_depth - 1, **kwargs)
                if not (len(candidate_tree) == 1 and "Other" in candidate_tree):
                    topic_tree[t]["sub-topics"] = candidate_tree
    return topic_tree


def cleanup(x: dict, min_topic_size: int = 5):
    x = deepcopy(x)
    # Collapse all the tiny topics into "Other"
    if "Other" not in x:
        x["Other"] = {"messages": []}

    keys = list(x.keys())
    for k in keys:
        if k == "Other":
            continue
        if len(x[k]["messages"]) < min_topic_size:
            tmp = x.pop(k)
            x["Other"]["messages"] += tmp["messages"]

    if not len(x["Other"]["messages"]):
        x.pop("Other")

    # TODO: Move all subtopics with just one topic one level up

    return x


def tree_summary(x: dict, include_messages: bool = True, min_topic_size: int = 1):
    x = cleanup(x, min_topic_size)
    out = {}
    for k, v in sorted(
        list(x.items()), key=lambda x: len(x[1]["messages"]), reverse=True
    ):
        m = v["messages"]
        new_k = f"{k} : {len(m)}"
        if "sub-topics" in v:
            out[new_k] = tree_summary(v["sub-topics"])
        else:
            out[new_k] = m if include_messages else ""

    if "Other" in out:
        # Move "other" to the end
        tmp = out.pop("Other")
        out["Other"] = tmp
        print("yay!")
    return out
