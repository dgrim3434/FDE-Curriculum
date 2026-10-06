import typing
from extraction.schemas import unwrap_optional, is_model

Dtype_mapping = {str: 'string', int: 'integer', float: 'number', bool: 'true or false'}


def type_name(annotation) -> str:
    
    core, nullable = unwrap_optional(annotation)
    
    if core in Dtype_mapping:
        
        words = Dtype_mapping.get(core)
    elif typing.get_origin(core) is typing.Literal:
        words = "one of: " + ", ".join(typing.get_args(core))
    elif typing.get_origin(core) is list:
        item = typing.get_args(core)[0]
        words = "list of " + item.__name__
    elif is_model(core):
        words = core.__name__
    else:
        words = str(core)
    
    if nullable:
        
        words += " or null"
    
    return words

def model_table(model):
    
    lines = [
        "## " + model.__name__,
        "| Field | Type | Required | Description |",
        "|---|---|---|---|",
    ]
    
    for name, field in model.model_fields.items():
        
        kind = type_name(field.annotation)
        if field.is_required():
            required = "yes"
        else:
            required = f"no (default: {field.default!r})"
        
        description = field.description or ""
        lines.append(f"| {name} | {kind} | {required} | {description} |")
    
    return "\n".join(lines)

def nested_models(model):
    
    found = []
    
    for name, field in model.model_fields.items():
        
        core, _ = unwrap_optional(field.annotation)
        
        if is_model(core):
            found.append(core)
        
        elif typing.get_origin(core) is list and is_model(typing.get_args(core)[0]):
            
            found.append(typing.get_args(core)[0])
    
    return found

def generate_docs(schema):
    
    tables = []
    to_do = [schema]
    done = set()
    
    while to_do:
        
        model = to_do.pop()
        
        if model in done:
            continue
        
        done.add(model)
        tables.append(model_table(model))
        to_do.extend(nested_models(model))
    
    return "\n\n".join(tables)

if __name__ == "__main__":
    from pathlib import Path
    from extraction.schemas import EntityTypes, Extraction, Event, Graph, EvidenceFirst, LabelFirst
    
    out = Path("docs/schemas.md")
    out.parent.mkdir(exist_ok=True)
    text = "\n\n".join(generate_docs(s) for s in [EntityTypes, Extraction, Event,Graph, EvidenceFirst, LabelFirst])

    out.write_text(text, encoding='utf-8')
    print("wrote", out)