from copy import deepcopy
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError, ConfigDict, Field, model_validator
from typing import Literal
import typing

class EvidenceFirst(BaseModel):
    model_config = ConfigDict(extra='forbid')
    evidence: str = Field(description="A short phrase copied exactly from the article that shows its topic")
    topic : Literal["World", "Sports", "Business", "Sci/Tech"] = Field(description="The article's topic")

class LabelFirst(BaseModel):
    model_config = ConfigDict(extra='forbid')
    topic : Literal["World", "Sports", "Business", "Sci/Tech"] = Field(description="The article's topic")
    evidence: str = Field(description="A short phrase copied exactly from the article that shows its topic")
    
class SearchInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    keyword: str = Field(description="The keyword which you would like to search for within the corpus.")
    limit: int = Field(default=5, ge=1, le=20, description="The number of sentences you would like to return that match the keyword you provided.")

class GetSentenceInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sentence_id: str = Field(description="The id of the sentence which you want to see.")

class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(description="The entity copied exactly from the sentence")
    type: Literal["PER", "ORG", "LOC", "MISC"] = Field(
        description="PER=person, ORG=organization, LOC=location, MISC=other incl. nationalities"
    )

class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[Entity] = Field(
        default=[], description="Every named entity in the sentence, one item per mention"
    )

class EntityTypes(BaseModel):
    model_config = ConfigDict(extra='forbid')
    has_person: bool = Field(description="Does the sentence contain a person")
    has_organization: bool = Field(description="Does the sentenct contain an organization")
    has_location: bool = Field(description="Does the sentence contain a location")
    has_misc: bool = Field(description="Does the sentence contain an entity which is not a person, organization, or location.")

class Location(BaseModel):
    model_config = ConfigDict(extra='forbid')
    city: str | None = Field(default=None, description="The city mentioned within the sentence. If no refrence exists the field shound be marked as None.")
    country: str | None = Field(default=None, description="The country of the mentioned location. If no refrence exists the field shound be marked as None.")

class Event(BaseModel):
    model_config = ConfigDict(extra='forbid')
    actor: str = Field(description="The entity which the sentence is refering to")
    action: str = Field(description="The action which the actor is taking")
    date: str | None = Field(default=None, description="The date of the action. If no date then mark as None")
    location: Location | None = Field(default=None, description="Where the event happened, or null if no place is mentioned.")

class RelEntity(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(description="The exact text of the object which is being refered to in the sentence")
    type: Literal["Peop", "Org", "Loc", "Other"] = Field(description="The kind of entity the text is. Peop means it's a person, Org means its an organization, Loc means it's a location. If it's none of those label it Other")

class Relation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    subject: str = Field(description="The exact text of the subject of the sentence")
    relation: Literal["Work_For", "Kill", "OrgBased_In", "Live_In", "Located_In"] = Field(description="The relationship between the subject and the object within the sentence")
    object: str = Field(description="The exact text of the object within the sentence")

class Graph(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    entities: list[RelEntity] = Field(default_factory=list, description="List of all of the entities mentioned in the sentence")
    relations: list[Relation] = Field(default_factory=list, description="List of all of the relations mentioned in the sentence")

    @model_validator(mode="after")
    def relations_point_to_entities(self):
        names = {e.text for e in self.entities}
        
        for i, r in enumerate(self.relations):
            for role in ("subject", "object"):
                value = getattr(r, role)
                
                if value not in names:
                    raise ValueError(f"relation {i + 1}: {role} {value!r} is not in the entities list")
        return self

NULL_LIKE = {"", "n/a", "na", "none", "null", "unknown"}

def wrap_list_field(data):
    
    if isinstance(data, dict):
        
        return [data], True
    return data, False

def normalize_enum(value, allowed):
    
    if value in allowed:
        return value, False
    if isinstance(value, str):
        
        for a in allowed:
            if value.strip().upper() == a.upper():
                return a, True
    
    return value, False

def nullify(value):
    
    if isinstance(value, str) and value.strip().lower() in NULL_LIKE:
        
        return None, True
    
    return value, False

@dataclass
class ValidationResult:
    ok: bool
    value: dict | None
    rules_fired: list
    errors: str | None

def is_model(t):
    
    return isinstance(t, type) and issubclass(t, BaseModel)

def unwrap_optional(annotation):
    
    args = typing.get_args(annotation)
    
    if type(None) in args:
        core = next(a for a in args if a is not type(None))
        
        return core, True
    
    return annotation, False

def walk(data, model_class, fired):
    
    for name, field in model_class.model_fields.items():
        
        if name not in data:
            continue
        
        value = data[name]
        
        core, nullable = unwrap_optional(field.annotation)
        
        if nullable:
            value, changed = nullify(value)
            
            if changed:
                
                fired.append("nullify")
        
        if typing.get_origin(core) is typing.Literal:
            value, changed = normalize_enum(value, typing.get_args(core))
            
            if changed:
                fired.append("normalize_enum")
        
        elif typing.get_origin(core) is list and is_model(typing.get_args(core)[0]):
            
            item_model = typing.get_args(core)[0]
            value, changed = wrap_list_field(value)
            
            if changed:
                
                fired.append("wrap_list_field")
            
            if isinstance(value, list):
                
                for item in value:
                    if isinstance(item, dict):
                        
                        walk(item, item_model, fired)
        
        elif is_model(core) and isinstance(value, dict):
            walk(value, core, fired)

        
        data[name] = value
        

def validate_with_repairs(data, schema):
    
    data_copy = deepcopy(data)
    
    fired = []

    if isinstance(data_copy, list):
        
        list_fields = [name for name, field in schema.model_fields.items() if typing.get_origin(unwrap_optional(field.annotation)[0]) is list]
        
        if len(list_fields) == 1:
            
            data_copy = {list_fields[0]: data_copy}
            fired.append("wrap_list_field")
        

    if isinstance(data_copy, dict):
        walk(data_copy, schema, fired)
    
    try:
        value = schema.model_validate(data_copy)
    
    except ValidationError as e:
        
        errors = e.errors(include_url=False, include_context=False)
        return ValidationResult(ok=False, value=None, rules_fired=fired, errors=errors)
    
    return ValidationResult(ok=True, value=value, rules_fired=fired, errors=[])

