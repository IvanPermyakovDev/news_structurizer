import Levenshtein


def wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
        
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
        
    distance = Levenshtein.distance(ref_words, hyp_words)

    return distance / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER)
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
        
    distance = Levenshtein.distance(reference, hypothesis)

    return distance / len(reference)


def mer(reference: str, hypothesis: str) -> float:
    """
    Calculate Match Error Rate (MER)
    """
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())
        
    if len(ref_words) == 0 and len(hyp_words) == 0:
        return 0.0
        
    matches = len(ref_words.intersection(hyp_words))
    total_words = len(ref_words) + len(hyp_words)
        
    if total_words == 0:
        return 0.0
            
    return 1.0 - (2 * matches) / total_words