from extraction.evaluation import to_pairs


def ungrounded(value, text):
    
    if value is None:
        return []
    
    entities = [e[0] for e in to_pairs(value)]
    ungrounded = []
    
    for entity in entities:
        #breakpoint()
        if entity not in text:
            ungrounded.append(entity)
    
    return ungrounded

def grounded_rate(records):
    
    total_entities = 0
    ungrounded_entities = 0
    
    for idx in range(len(records)):
        row = records[idx]
        
        if row['value'] is None:
            continue
        
        text = row['gold']['sentence']
        value = row['value']
        
        unground = ungrounded(value, text)
        
        total_entities += len(value['entities'])
        ungrounded_entities += len(unground)
    
    return (total_entities - ungrounded_entities) / total_entities if total_entities > 0 else None