"""
This parser file holds all of the parsing functions which will be needed throughout this week:
    - Parsing tags: <answer></answer>, <type></type> This function is left generic on purpose to ensure it can be used to extract data within any tags
    - ReAct outputs: Extracting test between Thought: ... and Action for the thought field and Text after Action for the action field with special guard rails to ensure we do not 
    include Observations: which the model might hallucinate on it's own
"""
from dataclasses import dataclass
import re
import string
from collections import Counter
from collections.abc import Callable
@dataclass
class ParseResults:
    ok: bool
    value: object | None
    error: str | None

FRAC_PATTERN = r"\b\d+/\d+\b"
"""
Parses the text using regex to find the text within the tags. If there are multiple tags we will take the last one
"""
def extract_tag(text: str, tag: str) -> str | None:
    
    pattern = rf"<{tag}>(.*?)</{tag}>"
    
    hits = re.findall(pattern, text, flags = re.IGNORECASE | re.S)
    
    if len(hits) == 0:
        return None
    
    return hits[-1]

def normalize_numbers(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    fracs = re.compile(FRAC_PATTERN)
    digits = re.compile(r"-?\d+(?:\.\d+)?")
    
    fracs = fracs.search(s)
    digits = digits.search(s)
    
    if fracs is None and digits is None:
        return None
    
    if fracs is None or fracs.start() > digits.start():
        
        return float(digits.group())
    
    frac = fracs.group()
    pos = frac.find("/")
    num = frac[:pos]
    den = frac[pos + 1:]
    
    if float(den) == 0:
        return None
    
    return float(num) / float(den)


def normalize_answer(s: str) -> str:
    
    s = s.lower()
    s = "".join([ch for ch in s if ch not in set(string.punctuation)])
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    
    return " ".join(s.split())

def numbers_equal(a, b) -> bool:
    
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6

def exact_match(pred, gold) -> bool:
    
    return normalize_answer(pred) == normalize_answer(gold)

def f1(pred, gold) -> float:
    
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if len(pred_norm) == 0 or len(gold_norm) == 0:
        return 0
    
    pred_list = pred_norm.split()
    gold_list = gold_norm.split()
    common = Counter(pred_list) & Counter(gold_list)
    
    n = sum(common.values())
    
    precision = n / len(pred_list)
    recall = n / len(gold_list)
    

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def normalize_label(s: str, labels: set[str]) -> str | None:
    
    norm_labels = {normalize_answer(s): s for s in labels}
    
    return norm_labels.get(normalize_answer(s))

def get_tag_error(tag: str) -> str:
    return f"Unable to parse the provided {tag}. Answer must be wrapped in <{tag}>YOUR ANSWER</{tag}> In order to be parsed. Try again with the corrected fromat"

def make_number_parser(tag: str = "answer") -> Callable[[str], ParseResults]:
    
    def get_results(s: str):
        
        value = extract_tag(text=s, tag = tag)
        
        if value is None:
            msg = get_tag_error(tag=tag)
            return ParseResults(ok=False, value=None, error=msg)
        
        number = normalize_numbers(value)
        
        if number is None:
            return ParseResults(ok=False, value=None, error=f"Tags where extracted but the result could not be converted into a float. Your answer must be numberic. Extracted response: {value}")
        
        return ParseResults(ok=True, value=number, error=None)
    
    return get_results



def make_label_parser(tag: str, allowed: set[str]) -> Callable[[str], ParseResults]:
    
    def get_results(s: str):
        
        label = extract_tag(text=s, tag=tag)
        
        if label is None:
            msg = get_tag_error(tag=tag)
            return ParseResults(ok=False, value=None, error=msg)
        
        ans = normalize_label(s=label, labels=allowed)
        
        if ans is None:
            return ParseResults(ok=False, value=None, error=f"Tags where extracted but the label did not match any of the labels within the provided options. Extracted label: {label}, Valid Options: {"|".join(allowed)}")

        return ParseResults(ok=True, value=ans, error=None)
    
    return get_results

def make_text_parser(tag: str = "answer") -> Callable[[str], ParseResults]:
    
    def get_results(s: str):
        
        text = extract_tag(text=s, tag=tag)
        
        if text is None:
            msg = get_tag_error(tag=tag)
            return ParseResults(ok=False, value=None, error=msg)
        
        if len(text.strip()) == 0:
            return ParseResults(ok=False, value=None, error="fThere was no text detected within the {tag}s. You must provide an answer.")
        return ParseResults(ok=True, value=text.strip(), error=None)
    
    return get_results


def make_pair_parser(tag1, tag2):
    
    p1, p2 = make_text_parser(tag1), make_text_parser(tag2)
    
    def parse(text):
        r1, r2 = p1(text), p2(text)
        
        if not r1.ok:
            return r1
        if not r2.ok:
            return r2
        
        return ParseResults(ok=True, value={tag1: r1.value, tag2: r2.value}, error=None)

    return parse
    
           
"""
React Parser:
    - Action: reasoning, function to call(must be within set), and args to the function
    - Finish: reasoning, answer
    - Parser Error  
"""

@dataclass
class Action:
    thought: str | None
    tool: str
    arg: str
    kept: str = ""

@dataclass
class Finish:
    thought: str | None
    answer: str
    kept: str = ""

@dataclass
class ReActParserError:
    reason: str
    raw: str
    kept: str = ""

"""
re.search returns the first match, or None. m.group(1) gives the first capture group. m.start() gives the position where the match begins, so text[:m.end()] keeps everything up to the end of the first Action line.
If the tool name is finish, return a Finish, not an Action.
"""

"""
The ReAct Format:
    <thought></thought>
    <action></action>
    <input></input>
    <finish></finish>
    <observation></observation> (The observation is the only thing which comes from code. Keeps the same format to decrease likelihood of hallucination on formatting)
"""
ACTION_EX = "if you want to search for John Doe your output format must be <action>search</action><input>John Doe</input>."
FINISH_EX = "If you have decide the final answer to the question is 5 then in order to answer your output must contain <finish>5</finish>"
def parse_react(text: str) -> Action | Finish | ReActParserError:
    
    action_call = extract_reAct(text=text, tag="action")
    finish_call = extract_reAct(text=text, tag='finish')
    # If no action call has been found then there must be a finish call else we have an error

    if (action_call is None) or (finish_call is not None and finish_call.start() < action_call.start()):
        
        if finish_call is None:
            return ReActParserError(reason=f"""The text must either contain an action call or a finish call neither where detected within the text. If you meant to do an action call
                                    your action must be wrapped in <action></action> tags. Example: <action>YOUR ACTION</action>. If you have reached your final answer and meant to write
                                    a finish call your answer must be wrapped in <finish></finish> tags like this <finish>YOUR ANSWER</finish>. Example: {FINISH_EX}""", raw=text, kept=text)
        
        finish_text = finish_call.group(1)
        if len(finish_text.strip()) == 0:
            return ReActParserError(reason=f"""Content within the finish block is empty. You must place a value inside of the finish block. Example: {FINISH_EX}""", raw=text, kept=text)
        
        thought_text = text[:finish_call.start()]
        
        return Finish(thought=extract_thought(thought_text), answer=finish_text.strip(), kept=text[:finish_call.end()])
    
    action_text = action_call.group(1)
    post_action_text = text[action_call.end():]
    input_call = extract_reAct(text=post_action_text, tag='input')
    
    if len(action_text.strip()) == 0:
        return ReActParserError(reason=f"""Contents within the action block is empty. The contents inside the action block must match one of the provided actions and cannot be empty. Example: {ACTION_EX}""",raw=text,kept=text)
    
    if input_call is  None:
        pre_action = text[:action_call.start()]
        inval_input = extract_reAct(pre_action, tag='input')
        
        if inval_input is None:
            return ReActParserError(reason=f"""An action call was detected but no input tags where found. The contents inside the action call specifies which function you want to call. Each action call must be immedidately followed
                                    up by it's corresponding input which must be placed inside of <input></input> tags. Example: {ACTION_EX}""", raw=text, kept=text[:action_call.end()])
    
        return ReActParserError(reason=f"""An action call was detected but no input tags where found following the <action> tag. Input tags where found before the action tags which is an invalid format.
                                Input tags placed before the action tag is an error and will be ignored by the parser. The format must be <action></action><input></input>. Example: {ACTION_EX}""", raw=text, kept=text[:action_call.end()])
    
        
    
    input_text = input_call.group(1)
    if len(input_text.strip()) == 0:
        return ReActParserError(reason=f"""The contents within the input tags was empty. This is an error, the contents within the input tags is the input to the function you specified and must not be empty. Example: {ACTION_EX}""", raw=text)
        
      
    between_action_tag = extract_reAct(text=post_action_text[:input_call.start()], tag='action')
    
    if between_action_tag is not None:
        return ReActParserError(reason=f"""Multiple Action tags where detected in the output. Your output must contain exactly one pair of action tags and one pair of input tags. The input
                                tags must directly follow the action tags. Example: {ACTION_EX}""",raw=text, kept="")
    
    
    thought_text = text[:action_call.start()]
    
    return Action(thought=extract_thought(thought_text), tool=action_text.strip(), arg=input_text.strip(), kept= get_kept(text))

def get_kept(text, input_tag = "input"):
    
    m = extract_reAct(text=text, tag=input_tag)    
    
    if m is None:
        return text
    
    return text[:m.end()]
        
    
def extract_thought(text, tag="thought") -> str | None:
    
    thought_call = extract_reAct(text=text, tag=tag)
    
    if thought_call is None:
        pattern = re.compile(rf"<{tag}>(.*?)", flags = re.I | re.S)
        val = pattern.search(text)
        if val is None:
            return text.strip() if len(text.strip()) > 0 else None
        
        return text[val.end():].strip() if len(text[val.end():].strip()) > 0 else None

    thought_text = thought_call.group(1)
    return thought_text.strip() if len(thought_text.strip()) > 0 else None
            


def extract_reAct(text: str, tag: str) -> str | None:
    
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", flags= re.I | re.S)
    
    val = pattern.search(text)
    
    return val