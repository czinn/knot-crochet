#!/usr/bin/python3

import argparse
import json
import math
import random
import itertools

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='Knot Crochet Pattern Generator',
            description='Generates patterns for crocheting knots')
    parser.add_argument('filename', help='Raw points for the torus')
    parser.add_argument('--tilt', '-t', type=int, help='Number of times longitude should wrap clockwise around the torus relative to the mesh longitude', default=0)
    parser.add_argument('--rows', '-r', type=int, help='Number of rows, i.e. the circumference of the torus', default=24)
    parser.add_argument('--gauge', '-g', type=float, help='Ratio of single crochet width over height', default=1.25)
    parser.add_argument('--stitches', '-s', help='Optional filename for raw stitch output')
    parser.add_argument('--meridian-fix', '-m', action='store_true', help='Swap meridian and longitude from default assignment')
    parser.add_argument('--flip', '-f', action='store_true', help='Flip the order in which the longitudes are traversed')
    parser.add_argument('--randomize-starts', action='store_true', help='Randomize where each row starts')
    parser.add_argument('--spiral', action='store_true', help='Make a spiral pattern instead of separate rows')

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

    if not args.spiral:
        stitches = make_stitches(points, args.rows, stitch_width, args.tilt, args.randomize_starts)
    else:
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
            stitch_for_prev = interpolate(stitch, next_stitch, 0.5) if args.spiral else stitch
            stitch_for_next = interpolate(stitch, prev_stitch, 0.5) if args.spiral else stitch
            prev_iter = circular_context(stitches[prev_row], last_in_prev, 5) # enumerate(stitches[prev_row])
            if args.spiral and j == len(row) - 1:
                prev_iter = itertools.chain(prev_iter, [(-1, row[0])])
            next_iter = circular_context(stitches[next_row], last_in_next, 5) # enumerate(stitches[next_row])
            if args.spiral and j == 0:
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

    print('total stitches: {}'.format(sum(len(row) for row in stitches)))
    print()
    print('chain {}'.format(len(stitches[-1])))
    any_last_in_first = False
    for i, row in enumerate(stitches):
        # print(connected_stitches[i])
        prev_row = (i - 1) % len(stitches)
        next_row = (i + 1) % len(stitches)
        normals = 0
        row_instructions = []
        net_increases = 0
        for j, stitch in enumerate(row):
            decreases = len(connected_stitches[i][j]) - 1
            is_decrease = decreases > 0
            increases = len(connected_stitches[i][j] & connected_stitches[i][j - 1])
            if j == 0 and -1 in connected_stitches[prev_row][-1] and 0 in connected_stitches[i][j]:
                increases += 1
            is_increase = increases > 0
            net_increases += increases - decreases
            # assert(not (is_decrease and is_increase))
            if is_increase or is_decrease:
                if normals > 0:
                    row_instructions.append(str(normals))
                    normals = 0
            if is_increase and is_decrease:
                if increases == 1 and decreases == 1:
                    normals += 1 # basically the same
                else:
                    row_instructions.append('id({}, {})'.format(increases, decreases))
            elif is_increase:
                if increases == 1:
                    row_instructions.append('inc')
                else:
                    row_instructions.append('inc{}'.format(increases))
            elif is_decrease:
                if decreases == 1:
                    row_instructions.append('dec')
                else:
                    row_instructions.append('dec{}'.format(decreases))
            else:
                normals += 1
        if normals > 0:
            row_instructions.append(str(normals))
            normals = 0
        last_in_first = False
        if -1 in connected_stitches[i][-1]:
            last_in_first = True # last stitch goes into first stitch in same row
            any_last_in_first = True
            net_increases += 1
        if -1 in connected_stitches[prev_row][-1]:
            net_increases -= 1
        print('row {} ({}{}): {}'.format(i + 1, len(row), '*' if last_in_first else '', ', '.join(row_instructions)))
        if net_increases != len(row) - len(stitches[prev_row]):
            print(connected_stitches[i])
            print('ERROR: Expected {} increases from previous row but found {}; try more rows'.format(len(row) - len(stitches[prev_row]), net_increases))
    if any_last_in_first:
        print()
        print('* Last stitch into first stitch in same row')
