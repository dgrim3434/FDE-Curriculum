from templates.template import render
from extraction.schemas import EntityTypes, Extraction, Event, Graph, EvidenceFirst, LabelFirst
from extraction.docs_gen import generate_docs
import json

EXAMPLES = {
    Extraction: {
        "text": "Reuters reported from Paris on Monday.",
        "output": {"entities": [{"text": "Reuters", "type": "ORG"},
                                {"text": "Paris", "type": "LOC"}]},
    },
    EntityTypes: {
        "text": "Reuters reported from Paris on Monday.",
        "output": {'has_person': False, 'has_organization': True, 'has_location': True, 'has_misc': False}
    },
    Event : {
        "text": "Microsoft announced a new office in Dublin.",
        "output": {"actor": 'Microsoft', "action": "announced a new office", "date": None, "location": {"city": "Dublin", "country": None}}
    },
    Graph : {
        "text": "Peter Blackburn reports for Reuters from Brussels.",
        "output": {"entities" : [{"text": "Peter Blackburn", "type": "Peop"}, {"text": "Reuters", "type": "Org"}, {"text": "Brussels", "type": "Loc"}],
                   "relations": [{"subject": "Peter Blackburn", "relation": "Work_For", "object": "Reuters"}]}
    },
    EvidenceFirst : {
        "text": "The Boston Red Sox failed to beat the yankees for the second year in a row in the wildcard round.",
        "output": {"evidence": "The Boston Red Sox", "topic": "Sports"}
    },
    LabelFirst : {
        "text": "The Boston Red Sox failed to beat the yankees for the second year in a row in the wildcard round.",
        "output": {"topic": "Sports", "evidence": "The Boston Red Sox"}
    }
    
}

def build_prompt(text, schema):
    layout = generate_docs(schema)
    example = EXAMPLES.get(schema)
    
    if example is None:
        raise ValueError(f"No example written for {schema.__name__}")
    
    output = json.dumps(example['output'])
    system_prompt = render(f'system/extraction.j2', layout=layout, text=example['text'], output=output)
    user_prompt = render(f'user/extraction.j2', text=text)
    
    return system_prompt, user_prompt
