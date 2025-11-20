# BoolQ prompts

def boolq_zero_shot(question, passage, model_type="decoder_only"):
    return (
        f"Passage: {passage}\n\n"
        f"Question: {question}\n"
        "Answer with Yes or No.\n"
        "Answer:"
    )


def boolq_one_shot(question, passage, model_type="decoder_only"):
    example = (
        "Passage: The sky appears blue because short wavelengths scatter more.\n"
        "Question: Is the sky blue because of Rayleigh scattering?\n"
        "Answer: Yes\n\n"
    )
    return (
        example +
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def boolq_cot(question, passage, model_type="decoder_only"):
    return (
        f"Passage: {passage}\n\n"
        f"Question: {question}\n"
        "Think step by step, then answer Yes or No.\n"
        "Reasoning:"
    )


# GSM8K prompts

def gsm8k_zero_shot(question, model_type="decoder_only"):
    return (
        "Solve the following math word problem. "
        "Give ONLY the final numeric answer.\n\n"
        f"Problem: {question}\n"
        "Answer:"
    )


def gsm8k_cot(question, model_type="decoder_only"):
    return (
        "Solve the following math word problem. "
        "Think step by step, and then give the final numeric answer.\n\n"
        f"Problem: {question}\n"
        "Reasoning:"
    )


# XSum prompts

def xsum_concise(document, model_type="encoder_decoder"):
    return (
        "Summarize the following article in one sentence.\n\n"
        f"{document}\n\nSummary:"
    )


def xsum_verbose(document, model_type="encoder_decoder"):
    return (
        "You are a highly capable text summarization system. "
        "Write a clear, concise, single-sentence abstractive summary that captures "
        "the essential information while avoiding unnecessary detail.\n\n"
        f"Article:\n{document}\n\nSummary:"
    )