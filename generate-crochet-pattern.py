#!/usr/bin/python3

import argparse
import json
import math
import random
import itertools
import unittest
import colorama
from llist import dllist

def dist(a, b):
    return sum((a - b)**2 for a, b in zip(a, b))**0.5

def interpolate(a, b, frac):
    return tuple(a * (1 - frac) + b * frac for a, b in zip(a, b))

def get(points, row, col):
    row, row_frac = math.floor(row), row - math.floor(row)
    col, col_frac = math.floor(col), col - math.floor(col)
    rows = len(points)
    cols = len(points[0])
    a = points[row % rows][col % cols]
    b = points[row % rows][(col + 1) % cols]
    c = points[(row + 1) % rows][col % cols]
    d = points[(row + 1) % rows][(col + 1) % cols]
    e = interpolate(a, b, col_frac)
    f = interpolate(c, d, col_frac)
    return interpolate(e, f, row_frac)

def shift_by(points, longitude_shift, shift_amount, grid_pos):
    pos = get(points, *grid_pos)
    while shift_amount > 0:
        col_shift = math.floor(grid_pos[1]) + 1 - grid_pos[1]
        next_grid_pos = (grid_pos[0] + col_shift * longitude_shift, grid_pos[1] + col_shift)
        next_pos = get(points, *next_grid_pos)
        d = dist(pos, next_pos)
        if d <= shift_amount:
            grid_pos = next_grid_pos
            pos = next_pos
            shift_amount -= d
        else:
            grid_pos = interpolate(grid_pos, next_grid_pos, shift_amount / d)
            pos = get(points, *grid_pos)
            break
    return (grid_pos, pos)

# deprecated
def make_stitches(points, rows, stitch_width, tilt=0, randomize_starts=False):
    longitude_shift = len(points) / len(points[0]) * tilt

    stitches = []
    for row_number in range(rows):
        grid_pos = (row_number * len(points) / rows, 0)
        pos = get(points, *grid_pos)
        row_stitches = []

        # calculate row length
        row_length = 0
        for i in range(len(points[0])):
            a = get(points, grid_pos[0] + i * longitude_shift, i)
            b = get(points, grid_pos[0] + (i + 1) * longitude_shift, i + 1)
            row_length += dist(a, b)

        target_stitches = math.floor(row_length / stitch_width + 0.5)
        row_stitch_width = row_length / target_stitches

        if randomize_starts:
            grid_pos, pos = shift_by(points, longitude_shift, row_stitch_width * random.random(), grid_pos)

        while True:
            row_stitches.append(pos)
            if len(row_stitches) == target_stitches:
                break
            grid_pos, pos = shift_by(points, longitude_shift, row_stitch_width, grid_pos)

        # sanity check the stitch lengths
        # print(row_stitch_width - dist(row_stitches[0], row_stitches[-1]), row_stitch_width)

        stitches.append(row_stitches)
    return stitches

def make_spiral_stitches(points, rows, stitch_width, tilt=0):
    grid_row_len = len(points[0])
    longitude_shift = (len(points) * (tilt + 1 / rows)) / len(points[0])

    full_spiral_length = 0
    for i in range(rows):
        row_start = i * len(points) / rows
        for j in range(grid_row_len):
            a = get(points, row_start + j * longitude_shift, j)
            b = get(points, row_start + (j + 1) * longitude_shift, j + 1)
            full_spiral_length += dist(a, b)

    target_stitches = math.floor(full_spiral_length / stitch_width + 0.5)
    actual_stitch_width = full_spiral_length / target_stitches

    grid_pos = (0, 0)
    pos = get(points, *grid_pos)
    stitches = []
    row_stitches = []
    for i in range(target_stitches):
        row_stitches.append(pos)
        grid_pos, pos = shift_by(points, longitude_shift, actual_stitch_width, grid_pos)
        new_row = False
        while grid_pos[1] > len(points[0]):
            new_row = True
            grid_pos = (grid_pos[0], grid_pos[1] - len(points[0]))
        if new_row:
            stitches.append(row_stitches)
            row_stitches = []
    if len(row_stitches) > 0:
        stitches.append(row_stitches)
    return stitches

# gets the c elements on either side of l[i] in l (plus l[i] itself)
def circular_context(l, i, c):
    start = i - c
    end = i + c + 1
    start_wrap = False
    end_wrap = False
    if start < 0:
        start += len(l)
        start_wrap = True
    if end >= len(l):
        end -= len(l)
        end_wrap = True
    if start_wrap and end_wrap:
        return enumerate(l)
    elif start_wrap or end_wrap:
        return itertools.chain(enumerate(l[start:], start=start), enumerate(l[:end]))
    else:
        return itertools.chain(enumerate(l[start:end], start=start))

def shortest_gap_between_same_stitch(shape_chain):
    shortest_gap = len(shape_chain)
    gap = 0
    last_special = None
    for i, s in enumerate(shape_chain):
        if s == 'n':
            gap += 1
        else:
            if s == last_special:
                shortest_gap = min(gap, shortest_gap)
            last_special = s
            gap = 0
    start_gap = next(i for i, s in enumerate(shape_chain) if s != 'n')
    if start_gap == last_special:
        shortest_gap = min(start_gap + gap, shortest_gap)

    return shortest_gap

"""
`shape_chain` is a list of stitches, notated as 'n' for normal, 'i' for increase, and 'd' for decrease, and viewed as a loop that is joined at the ends. `simplify(shape_chain)` combines nearby increase/decrease pairs such that there are at least `min_gap` normal stitches between each increase or decrease.
"""
def simplify(shape_chain, min_gap=3):
    if all(s == 'n' for s in shape_chain):
        return shape_chain

    # find an existing gap of at least `min_gap`, and rotate the string so it's at the start
    gap_size = 0
    gap_start = 0
    for i, s in enumerate(shape_chain):
        if s == 'n':
            gap_size += 1
            if gap_size >= min_gap:
                break
        else:
            gap_size = 0
            gap_start = i + 1
    if gap_size < min_gap:
        # look for a gap that wraps around the start and end of the string
        start_gap = next(i for i, s in enumerate(shape_chain) if s != 'n')
        end_gap = next(i for i, s in enumerate(reversed(shape_chain)) if s != 'n')
        if start_gap + end_gap >= min_gap:
            gap_start = len(shape_chain) - end_gap
            gap_size = start_gap + end_gap
    if gap_size < min_gap:
        # can't work without a gap of the right size
        return shape_chain

    rotated_chain = shape_chain[gap_start:] + shape_chain[:gap_start]

    i = 0
    while i < len(rotated_chain):
        # advance to the next special stitch
        i = next((i for i in range(i, len(rotated_chain)) if rotated_chain[i] != 'n'), len(rotated_chain))
        if i == len(rotated_chain):
            break

        gap = 0
        end = len(rotated_chain)
        last_special = rotated_chain[i]
        num_special = 1
        for j in range(i + 1, len(rotated_chain)):
            if rotated_chain[j] == 'n':
                gap += 1
                if gap >= min_gap:
                    end = j + 1
                    break
            else:
                if rotated_chain[j] == last_special:
                    end = j
                    break
                else:
                    last_special = rotated_chain[j]
                    num_special += 1
                gap = 0
        end = end - gap
        original_start = rotated_chain[i]
        for j in range(i, end):
            rotated_chain[j] = 'n'
        if num_special % 2 == 1:
            rotated_chain[(end - 1 - i) // 2 + i] = original_start

        i = end

    pivot = len(rotated_chain) - gap_start
    return rotated_chain[pivot:] + rotated_chain[:pivot]

class SimplifyTest(unittest.TestCase):
    def check_simplify(self, shape_chain, expected, min_gap=3):
        self.assertEqual(''.join(simplify(list(shape_chain), min_gap=3)), expected)

    def test_basic(self):
        self.check_simplify('nnnindnnn', 'nnnnnnnnn')

    def test_triple(self):
        self.check_simplify('nnnindninnn', 'nnnnninnnnn')

    def test_even_length(self):
        self.check_simplify('nnnindinn', 'nnnninnnn')

    def test_multiple(self):
        self.check_simplify('nnnindninnnidnn', 'nnnnninnnnnnnnn')

    def test_repeat(self):
        self.check_simplify('nnnininnn', 'nnnininnn')

    def test_repeat_complex(self):
        self.check_simplify('nnnidndnnn', 'nnnnnndnnn')

def reduce_runs(stitches_shape, last_in_first):
    all_stitches = list(itertools.chain.from_iterable(stitches_shape))
    original_gap_scale = shortest_gap_between_same_stitch(all_stitches)

    def get_stitch(i):
        return all_stitches[i % len(all_stitches)]

    stitches_since_last_change = 0
    def shift_stitch(i, di):
        nonlocal stitches_since_last_change
        nonlocal all_stitches
        i = i % len(all_stitches)
        ti = (i + di) % len(all_stitches)
        all_stitches[i], all_stitches[ti] = all_stitches[ti], all_stitches[i]
        stitches_since_last_change = 0

    def dist_to_next_special(i, di):
        d = 0
        while True:
            i += di
            d += 1
            if get_stitch(i) != 'n':
                return d

    def context(i, n=20):
        return ''.join(get_stitch(j) for j in range(i - n, i + n))

    prev_row_stitch = 0
    current_stitch = len(stitches_shape[0]) - (1 if last_in_first[0] else 0)

    def print_context():
        print(prev_row_stitch % len(all_stitches), current_stitch % len(all_stitches))
        print(context(prev_row_stitch))
        print(context(current_stitch))

    # TODO: The shifting is assymetric between increases and decreases. The
    # following pattern with increases is just the inverse of the pattern with
    # decreases, but the former is allowed while the latter would be shifted.
    # I think they should probably both be shifted.
    #
    # nnn n nn   nnnnnn
    # nnn ninn   nnnninn
    # nnninnnn   nnninnnn
    #
    # nnnnnnnn   nnnnnnnn
    # nnd nnnn   nndnnnn
    # nnn d nn   nnndnn
    #
    # Each increase or decrease forms a triangle, connecting two stitches in
    # one row with one stitch in another row. What this function should be
    # trying to do is shift some of the special stitches forward or back such
    # that there is a gap of at least one stitch between any two stitches which
    # are in a triangle.
    while stitches_since_last_change < len(all_stitches) or prev_row_stitch % len(all_stitches) != 0:
        stitch = get_stitch(current_stitch)
        if stitch == 'n':
            current_stitch += 1
            prev_row_stitch += 1
            stitches_since_last_change += 1
        elif stitch == 'i':
            # increase goes into previous stitch in previous row
            if get_stitch(prev_row_stitch - 1) != 'n':
                # we're going to shift last row stitch back and this stitch forward, or vice versa
                prev_back = dist_to_next_special(prev_row_stitch - 1, -1)
                prev_forward = dist_to_next_special(prev_row_stitch - 1, 1)
                cur_back = dist_to_next_special(current_stitch, -1)
                cur_forward = dist_to_next_special(current_stitch, 1)
                back_score = 1 / prev_forward + 1 / cur_back
                forward_score = 1 / prev_back + 1 / cur_forward
                if forward_score < back_score:
                    # shift cur forward and last row back
                    print('shifting this forward and last back')
                    print_context()
                    shift_stitch(current_stitch, 1)
                    shift_stitch(prev_row_stitch - 1, -1)
                else:
                    print('shifting this back and last forward')
                    print_context()
                    shift_stitch(current_stitch, -1)
                    shift_stitch(prev_row_stitch - 1, 1)
                    current_stitch -= 1
                    prev_row_stitch -= 1
            elif get_stitch(prev_row_stitch) != 'n':
                prev_forward = dist_to_next_special(prev_row_stitch, 1)
                cur_back = dist_to_next_special(current_stitch, -1)
                if cur_back >= prev_forward:
                    print('shifting this back')
                    print_context()
                    shift_stitch(current_stitch, -1)
                    current_stitch -= 1
                    prev_row_stitch -= 1
                else:
                    print('shift last forward')
                    print_context()
                    shift_stitch(prev_row_stitch, 1)
            elif get_stitch(prev_row_stitch - 2) != 'n':
                prev_back = dist_to_next_special(prev_row_stitch - 2, -1)
                cur_forward = dist_to_next_special(current_stitch, 1)
                if cur_forward >= prev_back:
                    print('shifting this forward')
                    print_context()
                    shift_stitch(current_stitch, 1)
                else:
                    print('shifting last back')
                    print_context()
                    shift_stitch(prev_row_stitch - 2, -1)
            else:
                current_stitch += 1
                # don't increase last row stitch since increase goes into previous stitch
                stitches_since_last_change += 1
        elif stitch == 'd':
            if get_stitch(prev_row_stitch) != 'n':
                print('d: shifting prev back and this forward')
                print_context()
                shift_stitch(prev_row_stitch, -1)
                shift_stitch(current_stitch, 1)
            elif get_stitch(prev_row_stitch + 1) != 'n':
                print('d: shifting prev forward and this back')
                print_context()
                shift_stitch(prev_row_stitch + 1, 1)
                shift_stitch(current_stitch, -1)
                current_stitch -= 1
                prev_row_stitch -= 1
            elif get_stitch(prev_row_stitch - 1) != 'n':
                print('d: shifting this forward')
                print_context()
                shift_stitch(current_stitch, 1)
            elif get_stitch(prev_row_stitch + 2) != 'n':
                print('d: shifting this back')
                print_context()
                shift_stitch(current_stitch, -1)
                current_stitch -= 1
                prev_row_stitch -= 1
            else:
                current_stitch += 1
                prev_row_stitch += 2
                stitches_since_last_change += 1
    first_row_len = current_stitch % len(all_stitches)

    new_gap_scale = shortest_gap_between_same_stitch(all_stitches)
    print('min gap changed from {} to {}'.format(original_gap_scale, new_gap_scale))

    all_stitches_iter = iter(all_stitches)
    stitches_shape = [[next(all_stitches_iter) for _ in row] for row in stitches_shape]
    return stitches_shape, first_row_len

def recompute_last_in_first(stitches_shape, first_row_len):
    last_in_first = [len(stitches_shape[0]) - first_row_len]
    for i in range(1, len(stitches_shape)):
        implied_prev_row_size = sum(1 if s == 'n' else 2 if s == 'd' else 0 for s in stitches_shape[i])
        actual_prev_row_size = len(stitches_shape[i - 1]) - last_in_first[i - 1]
        last_in_first.append(implied_prev_row_size - actual_prev_row_size)
    print(last_in_first)
    return last_in_first

def render_pattern(stitches_shape, last_in_first):
    stitches_shape = [r[:] for r in stitches_shape]
    stitches_shape.append(stitches_shape[0])
    last_in_first.append(last_in_first[0])
    columns = dllist()
    nodes = []

    def get_row_nodes(row, prev_row, start=0):
        row_nodes = []
        prev_row_index = start
        for s in row:
            prev_row_node = prev_row[prev_row_index] if prev_row_index < len(prev_row) else None
            if prev_row_node is None:
                prev_row_node = columns.append(None)
            if s == 'n':
                row_nodes.append(prev_row_node)
                prev_row_index += 1
            elif s == 'i':
                if prev_row_index == 0:
                    row_nodes.append(columns.appendleft(None))
                elif prev_row_index - 1 >= len(prev_row):
                    row_nodes.append(columns.append(None))
                else:
                    prev_row_node = prev_row[prev_row_index - 1]
                    if prev_row_node.next is not None and prev_row_node.next not in nodes[-1]:
                        row_nodes.append(prev_row_node.next)
                    else:
                        row_nodes.append(columns.insertafter(None, prev_row_node))
            elif s == 'd':
                row_nodes.append(prev_row_node)
                prev_row_index += 2
        return row_nodes

    wrapped_rows = []
    num_to_carry = 0
    carry_stitches = []
    carrying_decrease = False
    carry_counts = []
    for i, row in enumerate(stitches_shape):
        if last_in_first[i]:
            num_to_carry += 1
        carry_counts.append(len(carry_stitches))
        row, carry_stitches = carry_stitches + row, []
        num_carried = 0
        next_carrying_decrease = False
        while num_carried < num_to_carry:
            s = row.pop()
            if s == 'n':
                num_carried += 1
            elif s == 'd':
                num_carried += 2
                if num_carried > num_to_carry:
                    row.append(s)
                    next_carrying_decrease = True
                    break
            carry_stitches.append(s)
        carry_stitches.reverse()
        num_to_carry = len(carry_stitches)
        wrapped_rows.append(row[:])
        nodes.append(get_row_nodes(row, nodes[-1] if len(nodes) > 0 else [], start=1 if carrying_decrease else 0))
        carrying_decrease = next_carrying_decrease

    node = columns.first
    for i in range(len(columns)):
        node.value = i
        node = node.next

    for i, (row, row_nodes) in enumerate(zip(wrapped_rows, nodes)):
        row_str = ''
        stitches_in_row_so_far = 0
        for j, (s, n) in enumerate(zip(row, row_nodes)):
            if n.value > stitches_in_row_so_far:
                row_str += ' ' * (n.value - stitches_in_row_so_far)
                stitches_in_row_so_far = n.value
            stitch_str = s if s != 'n' else ','
            reset_style = False
            if s == 'i':
                stitch_str = colorama.Fore.GREEN + stitch_str
                reset_style = True
            if s == 'd':
                stitch_str = colorama.Fore.RED + stitch_str
                reset_style = True
            if carry_counts[i] == j:
                stitch_str = colorama.Style.BRIGHT + colorama.Back.CYAN + stitch_str
                reset_style = True
            if reset_style:
                stitch_str += colorama.Style.RESET_ALL
            row_str += stitch_str
            stitches_in_row_so_far += 1
        print(row_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='Knot Crochet Pattern Generator',
            description='Generates patterns for crocheting knots')
    parser.add_argument('filename', help='Raw points for the torus')
    parser.add_argument('--tilt', '-t', type=int, help='Number of times longitude should wrap clockwise around the torus relative to the mesh longitude', default=0)
    parser.add_argument('--rows', '-r', type=int, help='Number of rows, i.e. the circumference of the torus', default=24)
    parser.add_argument('--gauge', '-g', type=float, help='Ratio of single crochet width over height', default=1.25)
    parser.add_argument('--slant', type=float, help='Fraction of stitch that each stitch is behind the stitch it goes into in the previous row', default=0.25)
    parser.add_argument('--stitches', '-s', help='Optional filename for raw stitch output')
    parser.add_argument('--meridian-fix', '-m', action='store_true', help='Swap meridian and longitude from default assignment')
    parser.add_argument('--flip', '-f', action='store_true', help='Flip the order in which the longitudes are traversed')
    parser.add_argument('--randomize-starts', action='store_true', help='Randomize where each row starts')

    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        points = json.loads(f.read())

    if args.meridian_fix:
        points = list(map(list, zip(*points)))
    if args.flip:
        points = list(points[::-1])

    meridian_length = 0
    for i in range(len(points)):
        a = points[i - 1][0]
        b = points[i][0]
        meridian_length += dist(a, b)

    row_size = meridian_length / args.rows
    stitch_width = row_size * args.gauge

    stitches = make_spiral_stitches(points, args.rows, stitch_width, args.tilt)

    if args.stitches is not None:
        with open(args.stitches, 'w') as f:
            f.write(json.dumps(stitches))

    # find connections
    # connected_stitches gives the indices of the stitches in the previous row that each stitch is joined to
    connected_stitches = [[set() for _ in row] for row in stitches]
    for i, row in enumerate(stitches):
        prev_row = (i - 1) % len(stitches)
        next_row = (i + 1) % len(stitches)
        last_in_prev = 0
        last_in_next = 0
        for j, stitch in enumerate(row):
            prev_stitch = row[j - 1] if j > 0 else stitches[i - 1][-1] if i > 0 else stitches[-1][-1]
            next_stitch = row[j + 1] if j < len(row) - 1 else stitches[i + 1][0] if i < len(stitches) - 1 else stitches[0][0]
            stitch_for_prev = interpolate(stitch, next_stitch, args.slant)
            stitch_for_next = interpolate(stitch, prev_stitch, args.slant)
            prev_iter = circular_context(stitches[prev_row], last_in_prev, 5) # enumerate(stitches[prev_row])
            if j == len(row) - 1:
                prev_iter = itertools.chain(prev_iter, [(-1, row[0])])
            next_iter = circular_context(stitches[next_row], last_in_next, 5) # enumerate(stitches[next_row])
            if j == 0:
                next_iter = itertools.chain(next_iter, [(-1, row[-1])])
            closest_in_prev_row = min((dist(stitch_for_prev, other), k) for k, other in prev_iter)[1]
            closest_in_next_row = min((dist(stitch_for_next, other), k) for k, other in next_iter)[1]
            connected_stitches[i][j].add(closest_in_prev_row)
            last_in_prev = closest_in_prev_row
            last_in_next = closest_in_next_row
            if closest_in_next_row != -1:
                connected_stitches[next_row][closest_in_next_row].add(j)
            else:
                connected_stitches[i][-1].add(-1)

    stitches_shape = []
    for i, row in enumerate(stitches):
        row_shape = []
        prev_row = (i - 1) % len(stitches)
        next_row = (i + 1) % len(stitches)
        for j, stitch in enumerate(row):
            decreases = len(connected_stitches[i][j]) - 1
            is_decrease = decreases > 0
            increases = len(connected_stitches[i][j] & connected_stitches[i][j - 1])
            if j == 0 and -1 in connected_stitches[prev_row][-1] and 0 in connected_stitches[i][j]:
                increases += 1
            is_increase = increases > 0
            if is_increase and is_decrease:
                if increases == 1 and decreases == 1:
                    row_shape.append('n')
                elif increases - decreases == 1:
                    row_shape.append('i')
                elif decreases - increases == 1:
                    row_shape.append('d')
                else:
                    print('Unexpected increase-decrease combo: {} increases, {} decreases'.format(increases, decreases))
            elif is_increase:
                if increases == 1:
                    row_shape.append('i')
                else:
                    print('Unexpected multiple-increase ({})'.format(increases))
            elif is_decrease:
                if decreases == 1:
                    row_shape.append('d')
                else:
                    print('Unexpected multiple-decrease ({})'.format(decreases))
            else:
                row_shape.append('n')
        stitches_shape.append(row_shape)

    all_stitches = list(itertools.chain.from_iterable(stitches_shape))
    gap_scale = shortest_gap_between_same_stitch(all_stitches)
    all_stitches = simplify(all_stitches, min_gap=min(gap_scale, 5))
    all_stitches_iter = iter(all_stitches)
    stitches_shape = [[next(all_stitches_iter) for _ in row] for row in stitches_shape]

    last_in_first = [-1 in row[-1] for row in connected_stitches]

    #stitches_shape, first_row_len = reduce_runs(stitches_shape, last_in_first)
    #last_in_first = recompute_last_in_first(stitches_shape, first_row_len)

    if args.stitches is not None:
        with open(args.stitches, 'w') as f:
            output = [[{'s': s, 't': t} for s, t in zip(*x)] for x in zip(stitches, stitches_shape)]
            f.write(json.dumps(output))

    render_pattern(stitches_shape, last_in_first)

    print('total stitches: {}'.format(sum(len(row) for row in stitches)))
    print()
    print('chain {}'.format(len(stitches[-1])))

    any_last_in_first = False
    for i, row in enumerate(stitches_shape):
        row_instructions = []
        normals = 0
        net_increases = 0

        for stitch in row:
            if stitch != 'n':
                if normals > 0:
                    row_instructions.append(str(normals))
                    normals = 0

            if stitch == 'i':
                row_instructions.append('inc')
                net_increases += 1
            elif stitch == 'd':
                row_instructions.append('dec')
                net_increases -= 1
            elif stitch == 'n':
                normals += 1
            else:
                print('Unexpected stitch: {}'.format(stitch))
        if normals > 0:
            row_instructions.append(str(normals))
            normals = 0

        prev_row = (i - 1) % len(stitches)
        last_row_stitches = len(stitches[prev_row]) - last_in_first[prev_row]
        is_last_in_first = last_in_first[i] != 0
        if is_last_in_first:
            any_last_in_first = True
        print('row {} ({}{}): {}'.format(i + 1, len(row), '*' if is_last_in_first else '', ', '.join(row_instructions)))
        if net_increases != len(row) - last_in_first[i] - last_row_stitches:
            print(connected_stitches[i])
            print('ERROR: Expected {} increases from previous row but found {}; try more rows'.format(len(row) - last_in_first[i] - last_row_stitches, net_increases))
    if any_last_in_first:
        print()
        print('* Last stitch into first stitch in same row')
