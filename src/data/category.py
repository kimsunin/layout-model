"""
DocLayNet 카테고리와 인덱스를 매핑
"""

DOCLAYNET_CATEGORIES = [
    "Caption",         
    "Footnote",        
    "Formula",         
    "List-item",       
    "Page-footer",      
    "Page-header",      
    "Picture",          
    "Section-header",   
    "Table",            
    "Text",             
    "Title",            
]

RAW_TO_TYPE: dict[int, int | None] = {
    1: 0, 2: 1, 3: 2, 4: 3, 
    5: None, 
    6: None,
    7: 4, 8: 5, 9: 6, 10: 7, 11: 8,
}

NUM_TYPES = len({t for t in RAW_TO_TYPE.values() if t is not None})
PAD_TYPE = NUM_TYPES