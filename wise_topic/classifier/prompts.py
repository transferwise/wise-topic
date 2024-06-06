binary_prompt = """
{question}
Return the digit 1 for a positive answer, and 0 for a negative answer.
Return just the one character digit, nothing else.
Take a deep breath and think carefully before you make your reply. 
"""


def multi_choice_prompt(include_other: bool):
    out = (
        """I am about to give you a numbered list of options.
    Then I will pass to you a message (possibly, but not necessarily, a question), 
    after the word MESSAGE.
    Return an integer that is the number of the option that best fits that message,
    or if the message is a question, the number of the option that best answers the question.
    """
        + (
            """
    If no option fits the message, return 0.
    """
            if include_other
            else ""
        )
        + """
    Return only the number, without additional text.
    {categories}
    MESSAGE:
    {question}
    Take a deep breath and think carefully before you make your reply. 
    BEST MATCH OPTION NUMBER:"""
    )
    return out
