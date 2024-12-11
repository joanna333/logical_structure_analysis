from typing import Dict

LABEL_MAP: Dict[str, int] = {
    # Primary Relations
    'Explanation': 0,
    'Addition': 1,
    'Causal': 2,
    'Emphasis': 3,
    'Summary': 4,
    
    # Logical Relations
    'Conditional': 5,
    'Sequential': 6,
    'Comparison': 7,
    'Definition': 8,
    'Contrast': 9,
    
    # Elaborative Relations
    'Elaboration': 10,
    'Illustration': 11,
    'Concession': 12,
    'Generalization': 13,
    'Inference': 14,
    
    # Complex Relations
    'Problem Solution': 15,
    'Contrastive Emphasis': 16,
    'Purpose': 17,
    'Clarification': 18,
    'Enumeration': 19,
    'Cause and Effect': 20,
    'Temporal Sequence': 21
}

def get_label_name(label_id: int) -> str:
    """Get relation name from numeric label ID."""
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    return inv_map.get(label_id, "Unknown")

def validate_label(label: str) -> bool:
    """Validate if a label exists in the mapping."""
    return label in LABEL_MAP