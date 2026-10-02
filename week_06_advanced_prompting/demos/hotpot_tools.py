import difflib
import operator
import ast

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
MAX_EXPONENT = 100

def make_hotpot_tools(example) -> dict:
    
    titles = example['title']
    sentences = example['sentences']
    
    table = {titles[i]: sentences[i] for i in range(len(titles))}
    title_mapping = {titles[i].lower(): titles[i] for i in range(len(titles))}
    
    state = {"current": None, "seen": 0, 'last_keyword': None}
    
    def search(arg):
        norm_arg = arg.lower().strip()
        
        title = title_mapping.get(norm_arg)
        
        if title is None:
            title = difflib.get_close_matches(norm_arg, title_mapping.keys(), n=1, cutoff=0.5)
            
            if len(title) == 0:
                return f"No page found for {arg}. Available pages: {", ".join(title_mapping.values())}"
            
            title = title[-1]
            title = title_mapping[title]
            
            
        state['current'] = title
        state['seen'] = 0
        
        return f"{title}: " + "".join(table[title][:3])
    
    def lookup(arg):
        
        if state['current'] is None:
            return f"No page selected. You must use search to select a page first. Page Options: {", ".join(table.keys())}"
        
        norm_arg = arg.lower().strip()
        if len(norm_arg) == 0:
            return f"Provided lookup argument is empty. You must provide a word to lookup."
        
        if state['last_keyword'] == norm_arg:
            start = state['seen']
            state['last_keyword'] = norm_arg
        else:
            
            start = state['seen'] = 0
        
        sentences = table[state['current']]
        
        found = None
        
        for i, sent in enumerate(sentences[start:], start=start):
            
            
            if norm_arg in sent.lower():
                state['seen'] = i + 1
                found = sent.strip()
                break
        
        if found is None:
            return f"No results left of Page: {state['current']} for word: {norm_arg}. Try another keyword or search a different page."
        return found

    return {"search": search, "lookup": lookup, "calculate": calculate}


def _calc(node):
    
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        
        raise ValueError("Only numbers are allowed")
    
    if isinstance(node, ast.BinOp):
        op = OPS.get(type(node.op))
        if op is None:
            raise ValueError("Only + - * / ** are allowed")
        
        left = _calc(node.left)
        right = _calc(node.right)
        
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"Exponent larger than {MAX_EXPONENT} is not allowed")
        return op(left, right)
    
    if isinstance(node, ast.UnaryOp):
        value = _calc(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        
        raise ValueError("only - and + signs are allowed")
    
    raise ValueError("Only + - * / ** are allowed")


def calculate(arg: str) -> str:
    """Safe arithmetic tool. Always returns a string, never raises."""
    try:
        tree = ast.parse(arg.strip(), mode="eval")
        result = _calc(tree.body)
    except SyntaxError:
        return f"Invalid expression: {arg}"
    except ZeroDivisionError:
        return "Division by zero"
    except ValueError as e:
        return f"Not allowed: {e}"
    except OverflowError:
        return "Number too large"
    
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)