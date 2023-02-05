import bpy
import bmesh

import time
import json

mesh = bpy.context.active_object.data
bm = bmesh.new()
bm.from_mesh(mesh)

bm.verts.ensure_lookup_table()
vert = bm.verts[0]

def traverse_loop(loop):
    loop_list = [loop]
    while True:
        loop = loop.link_loop_radial_next.link_loop_next.link_loop_next
        if loop == loop_list[0]:
            return loop_list
        loop_list.append(loop)
        
meridian = traverse_loop(vert.link_loops[0])
other_loops = traverse_loop(vert.link_loops[1])
if len(meridian) > len(other_loops):
    meridian = other_loops

all_verts = []
for loop in meridian:
    all_verts.append([tuple(l.vert.co) for l in traverse_loop(loop.link_loop_prev)])

print(len(all_verts))
print(len(all_verts[0]))

# bpy.context.scene.cursor.location = all_verts[1][0]

with open(bpy.data.filepath.replace('blend', 'points.json'), 'w') as f:
    f.write(json.dumps(all_verts))