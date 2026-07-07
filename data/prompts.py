SYSTEM_TEMPLATE = (
    "A conversation between a User and an Assistant. The User poses a problem, "
    "the Assistant solves it. The Assistant first reasons through the problem "
    "step by step inside <think> </think> tags, then gives the final answer "
    "inside <answer> </answer> tags, with no text after the closing </answer> tag.\n\n"
    "User: {question}\nAssistant: <think>"
)


def build_prompt(question: str) -> str:
    """Wrap a raw question in the fixed think/answer instruction template.

    Note the trailing "<think>" — this forces the model to start its
    completion inside the think tag rather than needing to emit it itself,
    which stabilizes early-training format compliance.
    """
    return SYSTEM_TEMPLATE.format(question=question)
