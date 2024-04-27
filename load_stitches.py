import bpy
import bmesh

import json

with open(bpy.data.filepath.replace('blend', 'stitches.json'), 'r') as f:
    stitches = json.loads(f.read())

def make_color_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    mat.diffuse_color = color
    return mat

green = make_color_material('green', (0, 1, 0, 1))
red = make_color_material('red', (1, 0, 0, 1))

stitch_collection = bpy.data.collections.new('stitches')
bpy.context.scene.collection.children.link(stitch_collection)
for i, row in enumerate(stitches):
    for j, stitch in enumerate(row):
        if isinstance(stitch, list):
            stitch, type_ = stitch, 'n'
        else:
            stitch, type_ = stitch['s'], stitch['t']

        name = "stitch({},{})".format(i, j)
        mesh = bpy.data.meshes.new(name)
        sphere = bpy.data.objects.new(name, mesh)
        sphere.location = stitch
        stitch_collection.objects.link(sphere)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=4, v_segments=4, radius=0.07)
        bm.to_mesh(mesh)
        bm.free()

        if type_ == 'i':
            mesh.materials.append(green)
        elif type_ == 'd':
            mesh.materials.append(red)
