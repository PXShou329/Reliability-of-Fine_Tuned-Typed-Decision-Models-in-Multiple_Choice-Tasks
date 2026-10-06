"""Closed DSL; accepts operation names and bounded integer arrays only."""
PRIMITIVES=('ABS','ADD_ONE','DOUBLE','FILTER_EVEN','FILTER_POSITIVE','SORT_ASC','REVERSE','DEDUP_STABLE')
DESCRIPTIONS=('take absolute value of each element','add 1 to each element','multiply each element by 2','keep even elements','keep strictly positive elements','sort in ascending order','reverse element order','keep first occurrence of each value')
def run(program,array):
    if not isinstance(program,(list,tuple)) or len(program)>4 or any(type(x)!=str or x not in PRIMITIVES for x in program):raise ValueError('DSL program')
    if not isinstance(array,list) or len(array)>12 or any(type(x)!=int or abs(x)>1000000 for x in array):raise ValueError('array bound')
    a=array.copy()
    for op in program:
        if op=='ABS':a=[abs(x) for x in a]
        elif op=='ADD_ONE':a=[x+1 for x in a]
        elif op=='DOUBLE':a=[x*2 for x in a]
        elif op=='FILTER_EVEN':a=[x for x in a if x%2==0]
        elif op=='FILTER_POSITIVE':a=[x for x in a if x>0]
        elif op=='SORT_ASC':a=sorted(a)
        elif op=='REVERSE':a=a[::-1]
        elif op=='DEDUP_STABLE':a=list(dict.fromkeys(a))
        if any(abs(x)>1000000 for x in a):raise ValueError('numeric bound')
    return a
