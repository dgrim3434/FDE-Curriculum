from chunking.base import Chunk
from metadata.structure import section_path_at

def attach_metadata(chunks, doc_fields, sections) -> list[Chunk]:
    
    i = 0
    for i in range(len(chunks)):
        chunk = chunks[i]
        
        meta_data = {}
        meta_data['title'] = doc_fields['title']
        meta_data['authors'] = doc_fields['authors']
        meta_data['date'] = doc_fields['date']
        meta_data['provenance'] = doc_fields['provenance']
        
        meta_data['section_path'] = section_path_at(sections, chunk.start)
        meta_data['n_chunks'] = len(chunks)
        meta_data['prev_id'] = chunks[i -1].chunk_id if i > 0 else None
        meta_data['next_id'] = chunks[i +1].chunk_id if i < len(chunks) - 1 else None
    

        chunk.metadata = meta_data
        
    
    return chunks

def build_embed_text(chunk, mode) -> str:
    
    modes = set(['none', 'title', 'title_section'])
    
    if mode not in modes:
        raise ValueError(f"ERROR: invalid mode: {mode}. Mode options are: {" ,".join(modes)}")
    
    if mode == "none":
        
        return chunk.text
    if mode == "title":
        title = chunk.metadata['title']
        if title is None or len(title.strip()) == 0:
            return chunk.text
        
        return f"Title: {title}\n{chunk.text}"
    else:
        title = chunk.metadata['title']
        section_path = chunk.metadata['section_path']
        
        
        if title is None or len(title.strip()) == 0:
            title_string = ""
        else:
            title_string = f"Title: {title}\n"
        if section_path is None or len(section_path) == 0:
            section_string = ""
        else:
            section_string = " > ".join(section_path)
            
            section_string = "Section: " + section_string + "\n"
            
        return f"{title_string}{section_string}{chunk.text}"

def set_embed_text(chunks, mode) -> None:
    
    for chunk in chunks:
        
        chunk.metadata['embed_mode'] = mode
        
        chunk.metadata['embed_text'] = build_embed_text(chunk, mode)