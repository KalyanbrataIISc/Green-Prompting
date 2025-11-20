from datasets import load_dataset


def load_boolq(n: int = 200):
    """
    Load BoolQ validation subset of size n.
    """
    ds = load_dataset("google/boolq", split="validation")
    n = min(n, len(ds))
    return ds.shuffle(seed=42).select(range(n))


def load_gsm8k(n: int = 100):
    """
    Load GSM8K 'test' subset of size n.
    """
    ds = load_dataset("gsm8k", "main", split="test")
    n = min(n, len(ds))
    return ds.shuffle(seed=42).select(range(n))


def load_xsum(n: int = 200):
    """
    Load XSum 'test' subset of size n.

    For datasets==2.20.0, this script-based dataset still works.
    """
    ds = load_dataset("xsum", split="test")
    n = min(n, len(ds))
    return ds.shuffle(seed=42).select(range(n))