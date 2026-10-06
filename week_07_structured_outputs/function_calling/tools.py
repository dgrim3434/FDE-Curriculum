from data.loaders import load_conll2003
from extraction.schemas import Extraction, SearchInput, GetSentenceInput

data = load_conll2003(n=200)


def smoosh(text):
    return text.lower().strip()

def search_sentences(search_input: SearchInput) -> list:
    
    input = search_input.model_dump()
    keyword = input['keyword']
    limit = input['limit']
    results = []
    count = 0
    
    for idx in range(len(data)):
        
        id = data[idx]['id']
        text = data[idx]['sentence']
        
        if smoosh(keyword) in smoosh(text):
            
            count += 1
            results.append({'id': id, 'sentence': text})
            
            if count == limit:
                return results
    
    return results

def get_sentence(sentence_id: GetSentenceInput):
    
    input = sentence_id.model_dump()
    id = input['sentence_id']
    for idx in range(len(data)):
        
        if id == data[idx]['id']:
            return {'id': id, 'sentence': data[idx]['sentence']}
    
    raise ValueError(f"No sentence with ID: {id} was found within the corpus. Try a different ID or use a different tool.")

def record_entities(entities: Extraction) -> str:
    
    results = entities.model_dump()
    
    total_entities = len(results['entities'])
    
    return f"recorded: {total_entities} entities."


TOOL_MAPPING = {
    'search_sentences' : {'description': 'Takes a keyword and limit as an input. Searches over the entire corpus to find sentences which contain the keyword. The limit variable defines how many sentences get returned. Should be used when you need more information about a specific keyword.',
                          'input_model': SearchInput, 'function': search_sentences},
    'get_sentence': {'description': 'Takes a sentence ID as an input. If the ID corresponds to a sentence within the corpus it returns that sentence. Should be used when you know which sentence you need.',
                     'input_model': GetSentenceInput, 'function': get_sentence},
    'record_entities': {'description': 'Takes a list of entity mappings which maps the entity to its type. Should be used when you are ready to answer the question.',
                        'input_model': Extraction, 'function': record_entities}
}