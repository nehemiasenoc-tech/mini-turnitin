def calculate_lexical_similarity(tokens1: list[str], tokens2: list[str]) -> float:
    """
    Calcula la similitud léxica básica entre dos listas de tokens.
    Por ejemplo, usando la similitud de Jaccard.
    """
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    if not set1 and not set2:
        return 0.0
        
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union) * 100
