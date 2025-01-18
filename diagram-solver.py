from regina import *
from queue import Queue

# 7_6
# start_diagram = Link.fromPD([(3, 16, 4, 1), (1, 8, 2, 9), (9, 2, 10, 3), (15, 5, 16, 4), (12, 5, 13, 6), (6, 11, 7, 12), (10, 14, 11, 13), (14, 8, 15, 7)])
# target_diagram = Link.fromPD([(9, 1, 10, 16), (1, 4, 2, 5), (7, 2, 8, 3), (3, 8, 4, 9), (12, 5, 13, 6), (6, 13, 7, 14), (15, 11, 16, 10), (11, 15, 12, 14)])

# 7_7 (Link.fromDT([4, 8, 10, 12, 2, 14, 6]))

start_diagram = Link.fromPD([(1, 9, 2, 8), (7, 3, 8, 2), (3, 14, 4, 1), (11, 5, 12, 4), (5, 11, 6, 10), (13, 6, 14, 7), (9, 12, 10, 13)])
target_diagram = Link.fromPD([(5, 18, 6, 1), (1, 4, 2, 5), (2, 10, 3, 9), (10, 4, 11, 3), (6, 14, 7, 13), (14, 8, 15, 7), (17, 8, 18, 9), (11, 16, 12, 17), (15, 12, 16, 13)])

def track(diagram, op):
    crossings = [(i, diagram.crossing(i)) for i in range(diagram.size())]
    if op(diagram):
        return [(i, c.index()) for (i, c) in crossings if i != c.index()]
    return None

def adjacent_diagrams(diagram):
    ret = []
    # crossing moves
    for i in range(diagram.size()):
        d = Link(diagram)
        if (m := track(d, lambda d: d.r1(d.crossing(i)))) is not None:
            ret.append((('r1-', i, m), d))
        d = Link(diagram)
        if (m := track(d, lambda d: d.r2(d.crossing(i)))) is not None:
            ret.append((('r2-', i, m), d))
        for side in [0, 1]:
            d = Link(diagram)
            if (m := track(d, lambda d: d.r3(d.crossing(i), side))) is not None:
                ret.append((('r3', i, side, m), d))

        for strand in [0, 1]:
            # strand moves
            for side in [0, 1]:
                for sign in [-1, 1]:
                    d = Link(diagram)
                    if (m := track(d, lambda d: d.r1(d.crossing(i).strand(strand), side, sign))) is not None:
                        ret.append((('r1+', i, strand, side, sign, m), d))
                
                # r2+ moves
                # TODO: make this more efficient by walking the 2-cell instead of trying every possible strandref
                for other_crossing in range(diagram.size()):
                    for other_strand in [0, 1]:
                        for other_side in [0, 1]:
                            d = Link(diagram)
                            if (m := track(d, lambda d: d.r2(d.crossing(i).strand(strand), side, d.crossing(other_crossing).strand(other_strand), other_side))) is not None:
                                ret.append((('r2+', i, strand, side, other_crossing, other_strand, other_side, m), d))
    return ret

def bfs(start_diagram, target_diagram, max_additional_crossings=1):
    target_sig = target_diagram.knotSig(False, False)

    q = Queue()
    q.put(([], start_diagram))

    visited = set()

    max_crossings = start_diagram.size() + max_additional_crossings

    min_moves = None
    possible_paths = []
    while not q.empty():
        (path, d) = q.get()
        if min_moves is not None and len(path) > min_moves + 1:
            break
        if d.size() > max_crossings:
            continue
        sig = d.knotSig(False, False)
        if sig in visited:
            continue
        #print(sig)
        visited.add(sig)

        if sig == target_sig:
            if min_moves is None:
                min_moves = len(path)
            if len(path) > min_moves + 1:
                break
            possible_paths.append(path)
            print('found a possible path', path)

        for (move, new_d) in adjacent_diagrams(d):
            new_path = list(path)
            new_path.append(move)
            q.put((new_path, new_d))

    return possible_paths
