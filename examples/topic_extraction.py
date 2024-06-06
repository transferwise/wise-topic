from pprint import pprint

from langchain_openai import ChatOpenAI

from wise_topic import greedy_topic_tree, tree_summary

docs = [
    "The summer sun blazed high in the sky, bringing warmth to the sandy beaches.",
    "During summer, the days are long and the nights are warm and inviting.",
    "Ice cream sales soar as people seek relief from the summer heat.",
    "Families often choose summer for vacations to take advantage of the sunny weather.",
    "Many festivals and outdoor concerts are scheduled in the summer months.",
    "Winter brings the joy of snowfall and the excitement of skiing.",
    "The cold winter nights are perfect for sipping hot chocolate by the fire.",
    "Winter storms can transform the landscape into a snowy wonderland.",
    "Heating bills tend to rise as winter's chill sets in.",
    "Many animals hibernate or migrate to cope with the harsh winter conditions.",
    "Fish swim in schools to protect themselves from predators.",
    "Salmon migrate upstream during spawning season, a remarkable journey.",
    "Tropical fish add vibrant color and life to coral reefs.",
    "Overfishing threatens many species of fish with extinction.",
    "Fish have a diverse range of habitats from deep oceans to shallow streams.",
]

topic_llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)
# Do topic extraction on sampled data
topic_tree = greedy_topic_tree(
    docs,
    initial_topics=["Winter", "Summer"],
    topic_llm=topic_llm,
    max_depth=1,
    num_topics_per_update=1,
    max_unclassified_messages=2,
)

pprint(tree_summary(topic_tree))
pprint("**************")
pprint(topic_tree)
print("yay!")
