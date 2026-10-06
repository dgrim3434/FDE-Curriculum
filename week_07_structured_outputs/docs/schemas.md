## EntityTypes
| Field | Type | Required | Description |
|---|---|---|---|
| has_person | true or false | yes | Does the sentence contain a person |
| has_organization | true or false | yes | Does the sentenct contain an organization |
| has_location | true or false | yes | Does the sentence contain a location |
| has_misc | true or false | yes | Does the sentence contain an entity which is not a person, organization, or location. |

## Extraction
| Field | Type | Required | Description |
|---|---|---|---|
| entities | list of Entity | no (default: []) | Every named entity in the sentence, one item per mention |

## Entity
| Field | Type | Required | Description |
|---|---|---|---|
| text | string | yes | The entity copied exactly from the sentence |
| type | one of: PER, ORG, LOC, MISC | yes | PER=person, ORG=organization, LOC=location, MISC=other incl. nationalities |

## Event
| Field | Type | Required | Description |
|---|---|---|---|
| actor | string | yes | The entity which the sentence is refering to |
| action | string | yes | The action which the actor is taking |
| date | string or null | no (default: None) | The date of the action. If no date then mark as None |
| location | Location or null | no (default: None) | Where the event happened, or null if no place is mentioned. |

## Location
| Field | Type | Required | Description |
|---|---|---|---|
| city | string or null | no (default: None) | The city mentioned within the sentence. If no refrence exists the field shound be marked as None. |
| country | string or null | no (default: None) | The country of the mentioned location. If no refrence exists the field shound be marked as None. |

## Graph
| Field | Type | Required | Description |
|---|---|---|---|
| entities | list of RelEntity | no (default: PydanticUndefined) | List of all of the entities mentioned in the sentence |
| relations | list of Relation | no (default: PydanticUndefined) | List of all of the relations mentioned in the sentence |

## Relation
| Field | Type | Required | Description |
|---|---|---|---|
| subject | string | yes | The exact text of the subject of the sentence |
| relation | one of: Work_For, Kill, OrgBased_In, Live_In, Located_In | yes | The relationship between the subject and the object within the sentence |
| object | string | yes | The exact text of the object within the sentence |

## RelEntity
| Field | Type | Required | Description |
|---|---|---|---|
| text | string | yes | The exact text of the object which is being refered to in the sentence |
| type | one of: Peop, Org, Loc, Other | yes | The kind of entity the text is. Peop means it's a person, Org means its an organization, Loc means it's a location. If it's none of those label it Other |

## EvidenceFirst
| Field | Type | Required | Description |
|---|---|---|---|
| evidence | string | yes | A short phrase copied exactly from the article that shows its topic |
| topic | one of: World, Sports, Business, Sci/Tech | yes | The article's topic |

## LabelFirst
| Field | Type | Required | Description |
|---|---|---|---|
| topic | one of: World, Sports, Business, Sci/Tech | yes | The article's topic |
| evidence | string | yes | A short phrase copied exactly from the article that shows its topic |