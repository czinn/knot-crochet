import bpy
import bmesh

import json

with open(bpy.data.filepath.replace('blend', 'stitches.json'), 'r') as f:
    stitches = json.loads(f.read())

stitch_collection = bpy.data.collections.new('stitches')
bpy.context.scene.collection.children.link(stitch_collection)
for i, row in enumerate(stitches):
    for j, stitch in enumerate(row):
        name = "stitch({},{})".format(i, j)
        mesh = bpy.data.meshes.new(name)
        sphere = bpy.data.objects.new(name, mesh)
        sphere.location = stitch
        stitch_collection.objects.link(sphere)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=8, radius=0.07)
        bm.to_mesh(mesh)
        bm.free()