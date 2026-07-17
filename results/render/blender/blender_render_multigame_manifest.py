#!/usr/bin/env python3
"""
Blender-side renderer for MultiGame 16x16 fill-progress manifests.

Run through Blender, not system Python:
  blender --background --python scripts/blender_render_multigame_manifest.py -- --manifest manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import mathutils

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = REPO_ROOT / "assets"

ASSET_SETS = {
    "dungeon": {
        0: {"pack": "__procedural__", "file": "floor-stone-tiles", "scale": (1.0, 1.0, 1.0), "z": 0.0},
        1: {"pack": "mapped_blender_objects/dungeon/wall", "file": "wall.glb", "scale": (1.0, 1.0, 1.0), "z": 0.0},
        2: {"pack": "mapped_blender_objects/dungeon/interactable", "file": "interactable.glb", "scale": (0.9, 0.9, 0.9), "z": 0.02},
        3: {"pack": "__procedural__", "file": "monster-purple", "scale": (0.75, 0.75, 0.75), "z": 0.08},
        4: {"pack": "mapped_blender_objects/dungeon/collectable", "file": "collectable.glb", "scale": (0.72, 0.72, 0.72), "z": 0.02},
    },
    "sokoban": {
        0: {"pack": "mapped_blender_objects/sokoban/empty", "file": "empty.glb", "scale": (1.0, 1.0, 1.0), "z": 0.0},
        1: {"pack": "mapped_blender_objects/sokoban/wall", "file": "wall.glb", "scale": (1.0, 1.0, 1.0), "z": 0.0},
        2: {"pack": "__procedural__", "file": "push-block", "scale": (0.78, 0.78, 0.78), "z": 0.02},
        3: {"pack": "__procedural__", "file": "monster-purple", "scale": (0.75, 0.75, 0.75), "z": 0.08},
        4: {"pack": "mapped_blender_objects/sokoban/collectable", "file": "collectable.glb", "scale": (0.7, 0.7, 0.7), "z": 0.02},
    },
    "doom": {
        0: {"pack": "mapped_blender_objects/doom/empty", "file": "empty.glb", "scale": (1.0, 1.0, 1.0), "z": 0.0},
        1: {"pack": "mapped_blender_objects/doom/wall", "file": "wall.glb", "scale": (1.0, 1.0, 1.0), "z": 0.0},
        2: {"pack": "__procedural__", "file": "push-block", "scale": (0.78, 0.78, 0.78), "z": 0.02},
        3: {"pack": "__procedural__", "file": "monster-red", "scale": (0.75, 0.75, 0.75), "z": 0.08},
        4: {"pack": "mapped_blender_objects/doom/collectable", "file": "collectable.glb", "scale": (0.7, 0.7, 0.7), "z": 0.02},
    },
    "pokemon": {
        0: {"pack": "mapped_blender_objects/pokemon/empty", "file": "empty.glb", "scale": (1.0, 1.0, 0.26), "z": -0.08},
        1: {"pack": "mapped_blender_objects/pokemon/wall", "file": "wall.glb", "scale": (0.62, 0.62, 0.62), "z": 0.02},
        2: {"pack": "mapped_blender_objects/pokemon/interactable", "file": "interactable.glb", "scale": (0.9, 0.9, 0.9), "z": 0.02},
        3: {"pack": "__procedural__", "file": "monster-blue", "scale": (0.75, 0.75, 0.75), "z": 0.08},
        4: {"pack": "mapped_blender_objects/pokemon/collectable", "file": "collectable.glb", "scale": (0.104, 0.104, 0.104), "z": 0.50},
    },
    "zelda": {
        0: {"pack": "mapped_blender_objects/zelda/empty", "file": "empty.glb", "scale": (0.95, 0.95, 0.22), "z": -0.04},
        1: {"pack": "mapped_blender_objects/zelda/wall", "file": "wall.glb", "scale": (0.42, 0.42, 0.42), "z": 0.0},
        2: {"pack": "mapped_blender_objects/zelda/interactable", "file": "interactable.glb", "scale": (0.46, 0.46, 0.46), "z": 0.03},
        3: {"pack": "__procedural__", "file": "monster-red", "scale": (0.75, 0.75, 0.75), "z": 0.08},
        4: {"pack": "mapped_blender_objects/zelda/collectable", "file": "collectable.glb", "scale": (0.68, 0.68, 0.68), "z": 0.12},
    },
}

COLORS = {
    0: (0.48, 0.44, 0.36, 1.0),  # empty floor
    1: (0.34, 0.37, 0.40, 1.0),  # wall
    2: (0.10, 0.55, 0.28, 1.0),  # interactive
    3: (0.78, 0.18, 0.12, 1.0),  # hazard
    4: (0.95, 0.76, 0.12, 1.0),  # collectable
    "change_marker": (1.0, 0.82, 0.05, 1.0),
    "path_route": (1.0, 0.96, 0.04, 1.0),
    "path_start": (1.0, 0.96, 0.04, 1.0),
    "path_end": (1.0, 0.96, 0.04, 1.0),
    "path_shadow": (0.35, 0.26, 0.00, 1.0),
    "dark": (0.03, 0.035, 0.04, 1.0),
    "text": (0.9, 0.9, 0.88, 1.0),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    return parser.parse_args(argv)


def material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    if color[3] < 1.0:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = False
        mat.show_transparent_back = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Alpha"].default_value = color[3]
        bsdf.inputs["Roughness"].default_value = 0.62
    return mat


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def set_origin_mesh(obj: bpy.types.Object, mat: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.append(mat)
    return obj


def add_cube(name: str, loc: tuple[float, float, float], scale: tuple[float, float, float], mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return set_origin_mesh(obj, mat)


def add_cylinder(name: str, loc: tuple[float, float, float], mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.28, depth=0.95, location=loc)
    obj = bpy.context.object
    obj.name = name
    return set_origin_mesh(obj, mat)


def add_cone(name: str, loc: tuple[float, float, float], mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.42, radius2=0.0, depth=0.95, location=loc, rotation=(0, 0, math.radians(45)))
    obj = bpy.context.object
    obj.name = name
    return set_origin_mesh(obj, mat)


def add_sphere(name: str, loc: tuple[float, float, float], mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.28, location=loc)
    obj = bpy.context.object
    obj.name = name
    return set_origin_mesh(obj, mat)


def load_asset_collection(name: str, path: Path) -> bpy.types.Collection:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"No objects imported from {path}")

    collection = bpy.data.collections.new(f"asset_{name}")
    for obj in imported:
        obj.visible_shadow = True
        if hasattr(obj, "cycles_visibility"):
            obj.cycles_visibility.shadow = True
        for existing in list(obj.users_collection):
            existing.objects.unlink(obj)
        collection.objects.link(obj)
    if "pokemon_empty" in name:
        brighten_collection_materials(collection, amount=0.14)
    return collection


def brighten_collection_materials(collection: bpy.types.Collection, amount: float) -> None:
    seen: set[str] = set()
    for obj in collection.objects:
        data = getattr(obj, "data", None)
        materials = getattr(data, "materials", None)
        if not materials:
            continue
        for mat in materials:
            if mat is None or mat.name in seen:
                continue
            seen.add(mat.name)
            mat.diffuse_color = tuple(min(1.0, channel * 1.16 + amount * 0.25) for channel in mat.diffuse_color[:3]) + (mat.diffuse_color[3],)
            if not mat.use_nodes or not mat.node_tree:
                continue
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is None:
                continue
            base_color = bsdf.inputs.get("Base Color")
            if base_color is None:
                continue
            if base_color.links:
                old_link = base_color.links[0]
                source_socket = old_link.from_socket
                mat.node_tree.links.remove(old_link)
                bright = mat.node_tree.nodes.new("ShaderNodeBrightContrast")
                bright.name = "pokemon_empty_brightness"
                bright.inputs["Bright"].default_value = amount
                bright.inputs["Contrast"].default_value = 0.02
                mat.node_tree.links.new(source_socket, bright.inputs["Color"])
                mat.node_tree.links.new(bright.outputs["Color"], base_color)
            else:
                color = base_color.default_value
                base_color.default_value = (
                    min(1.0, color[0] * 1.16 + amount * 0.25),
                    min(1.0, color[1] * 1.16 + amount * 0.25),
                    min(1.0, color[2] * 1.16 + amount * 0.25),
                    color[3],
                )


def link_to_collection_only(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)


def make_procedural_monster_collection(name: str) -> bpy.types.Collection:
    colors = {
        "monster-purple": (0.32, 0.18, 0.72, 1.0),
        "monster-red": (0.72, 0.16, 0.12, 1.0),
        "monster-blue": (0.12, 0.34, 0.78, 1.0),
    }
    body_mat = material(f"{name}_body", colors.get(name, colors["monster-purple"]))
    horn_mat = material(f"{name}_horn", (0.08, 0.08, 0.09, 1.0))
    eye_mat = material(f"{name}_eye", (1.0, 0.92, 0.15, 1.0))
    pupil_mat = material(f"{name}_pupil", (0.02, 0.02, 0.02, 1.0))

    collection = bpy.data.collections.new(f"asset_procedural_{name}")

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.42, location=(0, 0, 0.48))
    body = bpy.context.object
    body.name = f"{name}_body"
    body.scale = (1.0, 0.82, 0.78)
    body.data.materials.append(body_mat)
    link_to_collection_only(body, collection)

    for x in (-0.18, 0.18):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.085, location=(x, -0.34, 0.56))
        eye = bpy.context.object
        eye.name = f"{name}_eye"
        eye.data.materials.append(eye_mat)
        link_to_collection_only(eye, collection)

        bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=0.035, location=(x, -0.405, 0.56))
        pupil = bpy.context.object
        pupil.name = f"{name}_pupil"
        pupil.data.materials.append(pupil_mat)
        link_to_collection_only(pupil, collection)

    for x, rot in ((-0.22, -18), (0.22, 18)):
        bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=0.085, radius2=0.0, depth=0.34, location=(x, 0.0, 0.88), rotation=(math.radians(rot), 0, 0))
        horn = bpy.context.object
        horn.name = f"{name}_horn"
        horn.data.materials.append(horn_mat)
        link_to_collection_only(horn, collection)

    for obj in collection.objects:
        obj.visible_shadow = True
        if hasattr(obj, "cycles_visibility"):
            obj.cycles_visibility.shadow = True
    return collection


def add_collection_cube(
    collection: bpy.types.Collection,
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    mat: bpy.types.Material,
) -> bpy.types.Object:
    obj = add_cube(name, loc, scale, mat)
    link_to_collection_only(obj, collection)
    return obj


def make_procedural_floor_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.new(f"asset_procedural_{name}")
    if name == "floor-metal-grid":
        base = material("floor_metal_base", (0.20, 0.22, 0.25, 1.0))
        plate = material("floor_metal_plate", (0.31, 0.34, 0.38, 1.0))
        line = material("floor_metal_line", (0.08, 0.09, 0.10, 1.0))
        bolt = material("floor_metal_bolt", (0.55, 0.57, 0.60, 1.0))
        add_collection_cube(collection, "metal_base", (0, 0, 0.015), (0.98, 0.98, 0.03), base)
        for x in (-0.24, 0.24):
            add_collection_cube(collection, "metal_plate", (x, 0, 0.042), (0.42, 0.88, 0.018), plate)
        add_collection_cube(collection, "metal_gap_x", (0, 0, 0.055), (0.055, 0.94, 0.012), line)
        add_collection_cube(collection, "metal_gap_y", (0, 0, 0.057), (0.94, 0.055, 0.012), line)
        for x in (-0.34, 0.34):
            for y in (-0.34, 0.34):
                bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=5, radius=0.045, location=(x, y, 0.075))
                obj = bpy.context.object
                obj.name = "metal_bolt"
                obj.scale.z = 0.18
                obj.data.materials.append(bolt)
                link_to_collection_only(obj, collection)
    elif name == "floor-stone-tiles":
        base = material("floor_stone_base", (0.38, 0.37, 0.36, 1.0))
        tile_a = material("floor_stone_tile_a", (0.47, 0.46, 0.43, 1.0))
        tile_b = material("floor_stone_tile_b", (0.32, 0.32, 0.31, 1.0))
        grout = material("floor_stone_grout", (0.13, 0.13, 0.14, 1.0))
        add_collection_cube(collection, "stone_base", (0, 0, 0.012), (0.99, 0.99, 0.024), base)
        for ix, x in enumerate((-0.25, 0.25)):
            for iy, y in enumerate((-0.25, 0.25)):
                mat = tile_a if (ix + iy) % 2 == 0 else tile_b
                z = 0.038 + (0.004 if ix == iy else 0.0)
                add_collection_cube(collection, "stone_tile", (x, y, z), (0.43, 0.43, 0.025), mat)
        add_collection_cube(collection, "stone_grout_x", (0, 0, 0.057), (0.055, 0.96, 0.012), grout)
        add_collection_cube(collection, "stone_grout_y", (0, 0, 0.058), (0.96, 0.055, 0.012), grout)
    else:
        raise ValueError(f"Unknown procedural floor asset: {name}")
    return collection


def make_procedural_push_block_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.new(f"asset_procedural_{name}")
    body = material("push_block_body", (0.92, 0.66, 0.16, 1.0))
    side = material("push_block_side", (0.54, 0.28, 0.08, 1.0))
    mark = material("push_block_mark", (0.18, 0.12, 0.08, 1.0))
    add_collection_cube(collection, "push_block_body", (0, 0, 0.38), (0.82, 0.82, 0.76), body)
    add_collection_cube(collection, "push_block_top", (0, 0, 0.78), (0.72, 0.72, 0.045), side)
    add_collection_cube(collection, "push_block_front_mark", (0, -0.415, 0.42), (0.42, 0.035, 0.28), mark)
    add_collection_cube(collection, "push_block_side_mark", (0.415, 0, 0.42), (0.035, 0.42, 0.28), mark)
    return collection


def make_procedural_collection(name: str) -> bpy.types.Collection:
    if name.startswith("monster-"):
        return make_procedural_monster_collection(name)
    if name.startswith("floor-"):
        return make_procedural_floor_collection(name)
    if name == "push-block":
        return make_procedural_push_block_collection(name)
    raise ValueError(f"Unknown procedural asset: {name}")


def asset_key(config: dict) -> str:
    return f"{config['pack']}/{config['file']}"


def load_asset_collections(asset_dir: Path) -> dict[str, bpy.types.Collection]:
    collections = {}
    for game, assets in ASSET_SETS.items():
        for category, config in assets.items():
            key = asset_key(config)
            if key in collections:
                continue
            if config["pack"] == "__procedural__":
                collections[key] = make_procedural_collection(config["file"])
                continue
            path = asset_dir / config["pack"] / config["file"]
            if not path.exists():
                raise FileNotFoundError(f"Missing asset for {game}:{category}: {path}")
            collections[key] = load_asset_collection(key.replace("/", "_"), path)
    return collections


def add_asset_instance(
    collection: bpy.types.Collection,
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.instance_type = "COLLECTION"
    obj.instance_collection = collection
    obj.location = loc
    obj.scale = scale
    obj.rotation_euler[2] = rotation_z
    bpy.context.collection.objects.link(obj)
    return obj


def stable_unit_float(*parts: object) -> float:
    key = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def stable_signed_float(*parts: object) -> float:
    return stable_unit_float(*parts) * 2.0 - 1.0


def object_jitter(game: str, category: int, row: int, col: int) -> tuple[float, float, float]:
    if game == "pokemon" and category == 1:
        amount = 0.11
    elif category == 1:
        amount = 0.06
    elif category in (2, 3, 4):
        amount = 0.22
    else:
        amount = 0.0
    dx = stable_signed_float(game, category, row, col, "x") * amount
    dy = stable_signed_float(game, category, row, col, "y") * amount
    if game == "pokemon" and category == 1:
        rot_amount = math.radians(180)
    else:
        rot_amount = math.radians(45 if category in (2, 3, 4) else 14)
    rot = stable_signed_float(game, category, row, col, "rot") * rot_amount
    if category == 1 and game != "pokemon":
        rot *= 0.35
    return dx, dy, rot


def object_scale(
    game: str,
    category: int,
    row: int,
    col: int,
    base_scale: tuple[float, float, float],
) -> tuple[float, float, float]:
    if game != "pokemon" or category != 1:
        return base_scale
    uniform = 0.90 + stable_unit_float(game, category, row, col, "scale") * 0.22
    height = 0.95 + stable_unit_float(game, category, row, col, "height") * 0.18
    return (base_scale[0] * uniform, base_scale[1] * uniform, base_scale[2] * uniform * height)


def add_text(text: str, loc: tuple[float, float, float], size: float, mat: bpy.types.Material) -> None:
    bpy.ops.object.text_add(location=loc, rotation=(math.radians(65), 0, 0))
    obj = bpy.context.object
    obj.name = "label"
    obj.data.body = text
    obj.data.align_x = "LEFT"
    obj.data.align_y = "CENTER"
    obj.data.size = size
    obj.data.materials.append(mat)


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    dx = target[0] - obj.location.x
    dy = target[1] - obj.location.y
    dz = target[2] - obj.location.z
    direction = mathutils.Vector((dx, dy, dz))
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_scene(width: int, height: int) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
        if hasattr(scene.eevee, "gtao_distance"):
            scene.eevee.gtao_distance = 4
        if hasattr(scene.eevee, "gtao_factor"):
            scene.eevee.gtao_factor = 1.6
        if hasattr(scene.eevee, "shadow_cube_size"):
            scene.eevee.shadow_cube_size = "2048"
        if hasattr(scene.eevee, "shadow_cascade_size"):
            scene.eevee.shadow_cascade_size = "2048"
    scene.world = bpy.data.worlds.new("world") if scene.world is None else scene.world
    scene.world.color = (0.08, 0.085, 0.09)

    bpy.ops.object.light_add(type="SUN", location=(12, -8, 32))
    sun = bpy.context.object
    sun.name = "sun_light"
    sun.rotation_euler = (math.radians(25), 0, math.radians(18))
    sun.data.energy = 2.8
    if hasattr(sun.data, "use_shadow"):
        sun.data.use_shadow = True

    bpy.ops.object.light_add(type="AREA", location=(7.5, -2, 24))
    light = bpy.context.object
    light.name = "large_key_light"
    light.data.energy = 230
    light.data.size = 14
    if hasattr(light.data, "use_shadow"):
        light.data.use_shadow = True

    bpy.ops.object.camera_add(location=(12.5, -9.5, 18.0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    target = (7.5, 7.5, 0.55)
    direction = mathutils.Vector((target[0] - camera.location.x, target[1] - camera.location.y, target[2] - camera.location.z))
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 16.6


def add_cylinder_between(
    name: str,
    start: mathutils.Vector,
    end: mathutils.Vector,
    radius: float,
    mat: bpy.types.Material,
) -> bpy.types.Object | None:
    direction = end - start
    if direction.length <= 1e-6:
        return None
    midpoint = (start + end) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=direction.length, location=midpoint)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    return obj


def add_stage(
    level: list[list[int]],
    game: str,
    mats: dict,
    asset_collections: dict[str, bpy.types.Collection] | None,
) -> None:
    board_x = 0.0
    board_y = 0.0
    asset_set = ASSET_SETS.get(game, ASSET_SETS["dungeon"])

    for row in range(16):
        for col in range(16):
            category = int(level[row][col])
            x = board_x + col
            y = board_y + (15 - row)
            if asset_collections:
                floor_cfg = asset_set[0]
                add_asset_instance(
                    asset_collections[asset_key(floor_cfg)],
                    "floor",
                    (x + 0.5, y + 0.5, floor_cfg["z"]),
                    floor_cfg["scale"],
                )
                if category != 0:
                    cfg = asset_set[category]
                    dx, dy, rotation_z = object_jitter(game, category, row, col)
                    scale = object_scale(game, category, row, col, cfg["scale"])
                    add_asset_instance(
                        asset_collections[asset_key(cfg)],
                        f"category_{category}",
                        (x + 0.5 + dx, y + 0.5 + dy, cfg["z"]),
                        scale,
                        rotation_z,
                    )
            else:
                add_cube("floor", (x + 0.5, y + 0.5, 0.025), (0.96, 0.96, 0.05), mats[0])
                if category == 1:
                    add_cube("wall", (x + 0.5, y + 0.5, 0.58), (0.86, 0.86, 1.08), mats[1])
                elif category == 2:
                    add_cylinder("interactive", (x + 0.5, y + 0.5, 0.55), mats[2])
                elif category == 3:
                    add_cone("hazard", (x + 0.5, y + 0.5, 0.56), mats[3])
                elif category == 4:
                    add_sphere("collectable", (x + 0.5, y + 0.5, 0.48), mats[4])


def add_change_markers(cells: list[list[int]], mats: dict) -> None:
    marker_mat = mats["change_marker"]
    for row, col in cells:
        x = col + 0.5
        y = 15 - row + 0.5
        z = 0.15
        length = 0.58
        thickness = 0.045
        add_cube("change_marker_x", (x, y - 0.43, z), (length, thickness, thickness), marker_mat)
        add_cube("change_marker_y", (x - 0.43, y, z), (thickness, length, thickness), marker_mat)
        add_cube("change_marker_x", (x, y + 0.43, z), (length, thickness, thickness), marker_mat)
        add_cube("change_marker_y", (x + 0.43, y, z), (thickness, length, thickness), marker_mat)


def add_path_route(coords: list[list[int]], mats: dict) -> None:
    if len(coords) < 2:
        return
    route_mat = mats["path_route"]
    shadow_mat = mats["path_shadow"]
    start_color = COLORS["path_start"]
    end_color = COLORS["path_end"]
    points = [
        mathutils.Vector((float(col) + 0.5, 15.0 - float(row) + 0.5, 1.18))
        for row, col in coords
    ]
    shadow_offset = mathutils.Vector((0.0, 0.0, -0.035))
    for start, end in zip(points, points[1:]):
        add_cylinder_between("path_shadow", start + shadow_offset, end + shadow_offset, 0.195, shadow_mat)
        add_cylinder_between("path_route", start, end, 0.155, route_mat)
    for index, (point, color) in enumerate(((points[0], start_color), (points[-1], end_color))):
        endpoint_mat = material(f"mat_path_endpoint_{index}", color)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.42, location=point)
        obj = bpy.context.object
        obj.name = "path_endpoint" if index == 0 else "path_endpoint_end"
        obj.data.materials.append(endpoint_mat)


def render_stage(
    stage: dict,
    resolution: tuple[int, int],
    asset_collections: dict[str, bpy.types.Collection] | None,
) -> None:
    clear_scene()
    setup_scene(*resolution)
    mats = {key: material(f"mat_{key}", color) for key, color in COLORS.items()}
    add_stage(
        stage["unified"],
        stage.get("game", "dungeon"),
        mats,
        asset_collections,
    )
    if stage.get("changed_cells"):
        add_change_markers(stage["changed_cells"], mats)
    if stage.get("path_coords"):
        add_path_route(stage["path_coords"], mats)

    out_path = Path(stage["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    print(f"saved {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    resolution = tuple(manifest.get("resolution", [2400, 760]))
    asset_collections = load_asset_collections(Path(args.asset_dir))
    stages = [
        stage
        for sample in manifest["samples"]
        for stage in sample["stages"]
    ]
    iterator = tqdm(stages, desc="render stages", unit="stage") if tqdm else stages
    total = len(stages)
    for index, stage in enumerate(iterator, start=1):
        if tqdm is None:
            print(
                f"render stages: {index}/{total} {stage.get('game', 'game')} {stage.get('label', '')}",
                flush=True,
            )
        render_stage(stage, resolution, asset_collections)


if __name__ == "__main__":
    main()
