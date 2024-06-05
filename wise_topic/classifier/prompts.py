binary_prompt = """
{question}
Return the digit 1 for a positive answer, and 0 for a negative answer.
Return just the one character digit, nothing else.
Take a deep breath and think carefully before you make your reply. 
"""

multi_choice_prompt = """
{question}
Choose the most suitable option from the following:
{numbered_options}
Return the number of the most suitable option.
Return just the one character digit, nothing else.
Take a deep breath and think carefully before you make your reply. 
"""
