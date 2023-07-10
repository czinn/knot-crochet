#!/usr/bin/python3

import argparse
import json
import numpy as np
import svgwrite

def proj(a, b):
    return np.dot(a, b) / np.dot(b, b) * b

def normalize(a):
    return a / np.linalg.norm(a)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='Knot Crochet Pattern Generator',
            description='Generates patterns for crocheting knots')
    parser.add_argument('filename', help='Stitches for the pattern')
    parser.add_argument('--rows', '-r', type=int, nargs='+', help='Which rows of stitches to project')
    parser.add_argument('--normal', '-n', type=float, nargs=3, help='Normal vector for the projection plane')
    parser.add_argument('--output', '-o', help='Output SVG file')

    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        stitches = json.loads(f.read())

    plane_normal = normalize(np.array(args.normal))
    def point_on_plane(point):
        return point - np.dot(point, plane_normal) * plane_normal

    plane_x = normalize(point_on_plane(np.array([1, 1, 1])))
    plane_y = normalize(np.cross(plane_normal, plane_x))

    def plane_coords(point):
        on_plane = point_on_plane(point)
        return np.array([np.dot(on_plane, plane_x), np.dot(on_plane, plane_y), np.dot(point, plane_normal)])

    projected_points = [[plane_coords(point) * 100 for point in stitches[row]] for row in args.rows]

    segments = [(-0.5 * (a[2] + b[2]), a[:2], b[:2], row_index) for row_index, row in enumerate(projected_points) for a, b in zip(row, row[1:] + [row[0]])] # Z, point 1, point 2, row
    segments.sort()

    strokes = ['black', 'red', 'blue']

    drawing = svgwrite.Drawing(args.output)
    xmin, xmax, ymin, ymax = segments[0][1][0], segments[0][1][0], segments[0][1][1], segments[0][1][1]
    for (_z, a, b, r) in segments:
        bg_line = drawing.add(svgwrite.shapes.Line(start=a, end=b))
        bg_line.stroke('white', width=7, linecap='butt')
        line = drawing.add(svgwrite.shapes.Line(start=a, end=b))
        line.stroke(strokes[r], width=3, linecap='round')
        xmin = min(a[0], xmin)
        xmax = max(a[0], xmax)
        ymin = min(a[1], ymin)
        ymax = max(a[1], ymax)

    width = xmax - xmin
    height = ymax - ymin
    xmin -= width * 0.1
    xmax += width * 0.1
    ymin -= height * 0.1
    ymax += height * 0.1
    drawing.update({'viewBox': ' '.join(map(str, (xmin, ymin, xmax - xmin, ymax - ymin)))})
    drawing.save()
