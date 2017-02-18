def _mirror_indices(indices, start_after, end_before):
    '''
    input: indices=[0, 1, 2, 3, 4, 5, 6, 7], start_after=2, end_before=6
    output: [0, 1, 2, 5, 4, 3, 6, 7]
    '''
    start_loc = indices.index(start_after)
    end_loc = indices.index(end_before)

    if start_loc >= end_loc:
    	raise ValueError("'start_after' must be to the left of 'end_before'")

    before = indices[:start_loc+1]
    middle = indices[start_loc+1:end_loc]
    after = indices[end_loc:]
    return before + middle[::-1] + after




ls = [0, 1, 2, 3, 4, 5, 6, 7]
print ls
print "++++++++++++++++"
print(_mirror_indices(ls, 2, 6))