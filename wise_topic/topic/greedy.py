from collections import defaultdict
from copy import deepcopy
from typing import Optional, Sequence, Callable, Union
import logging

import numpy as np
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from .classifier import topic_classifier
from .topic_extractor import extract_topics


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
        topic_llm: Union[str, BaseLanguageModel] = "gpt-4-turbo",
        classifier: Callable = topic_classifier,
        classifier_llm: Union[str, BaseLanguageModel] = "gpt-3.5-turbo",
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
        self.num_topics_per_pass = num_topics_per_update
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
        self.verbose = verbose

        self.topics = [] if initial_topics is None else list(initial_topics)
        self.messages_to_topics = {}
        self.unclassified_messages = []

    def __call__(self, messages: Sequence[str]):
        np.random.shuffle(messages)
        for i, m in enumerate(messages):
            self.step(m)
        logging.info(self.topic_counts)

    def step(self, message: str):
        if len(self.topics):
            topic_num = self.classifier(self.topics, message, llm=self.classifier_llm)
            assert topic_num <= len(self.topics)
        else:
            topic_num = 0

        if topic_num == 0:
            self.unclassified_messages.append(message)
        else:
            self.messages_to_topics[message] = topic_num - 1
            if self.verbose:
                print(message, "\n", self.topics[topic_num - 1], "\n***********")

        update_topics = False
        if len(self.topics):
            if len(self.unclassified_messages) > self.max_unclassified_messages:
                update_topics = True
        else:
            if len(self.unclassified_messages) > self.initial_unclassified_messages:
                update_topics = True

        if update_topics:
            logging.info("Updating topics...")
            self.update_topics()
            logging.info(self.topic_counts)

        return topic_num

    def update_topics(self):
        messages = self.unclassified_messages
        self.unclassified_messages = []

        new_topics = self.extract_topics(messages)
        self.topics += new_topics

        for m in messages:
            self.step(m)

    def extract_topics(self, messages: Sequence[str]):
        logging.info("updating topics...")

        num_topics = (
            self.num_topics_per_pass if len(self.topics) else self.initial_num_topics
        )

        topics = self.topic_extractor(
            messages,
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
    for key in ["max_unclassified_messages", "initial_unclassified_messages"]:
        if key in kwargs and kwargs[key] < 1:
            kwargs[key] = int(len(messages) * kwargs[key])

    gte = GreedyTopicExtractor(**kwargs)
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


def tree_summary(x: dict):
    x = deepcopy(x)
    for k, v in x.items():
        m = v.pop("messages")
        if "sub-topics" in v:
            x[k] = tree_summary(v["sub-topics"])
        else:
            x[k] = len(m)
    return x
