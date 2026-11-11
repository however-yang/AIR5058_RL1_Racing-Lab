"""MuJoCo visual geometry for the circuit and its GT car."""

from xml.etree import ElementTree as ET

import numpy as np


def numbers(values):
    return " ".join(f"{float(value):.7g}" for value in values)


def prop(parent, kind, pos, size, color, **attrs):
    return ET.SubElement(parent, "geom", type=kind, pos=numbers(pos), size=numbers(size),
                         rgba=color, contype="0", conaffinity="0", mass="0", **attrs)


def camera(parent, name, position, target, fovy):
    forward = np.asarray(target) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ET.SubElement(parent, "camera", name=name, pos=numbers(position),
                  xyaxes=numbers([*right, *up]), fovy=str(fovy))


def shell(assets, car, name, sections, color, material, top_scale=1.):
    vertices = [(x, y, z) for x, width, bottom, top in sections
                for y, z in [(-width, bottom), (width, bottom),
                             (width * top_scale, top), (-width * top_scale, top)]]
    last = len(vertices) - 4
    faces = [(0, 2, 1), (0, 3, 2), (last, last + 1, last + 2), (last, last + 2, last + 3)]
    for start in range(0, last, 4):
        for corner in range(4):
            a, b = start + corner, start + (corner + 1) % 4
            faces.extend([(a, b, b + 4), (a, b + 4, a + 4)])
    ET.SubElement(assets, "mesh", name=name, vertex=numbers(np.ravel(vertices)),
                  face=" ".join(str(index) for face in faces for index in face))
    ET.SubElement(car, "geom", type="mesh", mesh=name, rgba=color, material=material,
                  contype="0", conaffinity="0", mass="0")


def dress_car(assets, world):
    car = world.find("body[@name='car']")
    for geom in car.findall("geom"):
        geom.set("rgba", "0 0 0 0")
    ET.SubElement(assets, "material", name="paint", specular=".35", shininess=".42")
    ET.SubElement(assets, "material", name="glass", specular=".5", shininess=".7")
    aqua = ".015 .57 .62 1"
    orange = "1 .26 .055 1"
    graphite = ".045 .06 .073 1"
    ice = ".77 .96 1 1"
    shell(assets, car, "gt_body", [(-.278, .131, .015, .092), (-.220, .155, .010, .125),
                                 (.135, .158, .010, .118), (.225, .145, .018, .091),
                                 (.280, .122, .023, .072)], aqua, "paint")
    shell(assets, car, "gt_canopy", [(-.155, .112, .112, .126), (-.100, .091, .123, .208),
                                   (.014, .081, .123, .213), (.107, .105, .107, .122)],
          ".035 .10 .14 1", "glass", top_scale=.77)
    prop(car, "ellipsoid", (0., 0., -.077), (.296, .211, .002), ".012 .02 .025 .38")
    prop(car, "box", (-.045, 0, .213), (.047, .065, .003), aqua)
    prop(car, "box", (-.045, -.018, .217), (.048, .012, .002), orange)
    prop(car, "box", (.18, -.018, .105), (.047, .013, .002), orange, euler="0 .30 0")
    prop(car, "box", (.269, 0, .022), (.025, .143, .007), graphite)
    prop(car, "box", (.281, 0, .047), (.007, .067, .013), graphite)
    prop(car, "box", (-.241, 0, .170), (.043, .177, .006), graphite, euler="0 -.11 0")
    for side in (-1, 1):
        prop(car, "box", (-.238, .109 * side, .134), (.009, .009, .036), graphite)
        prop(car, "box", (-.248, .178 * side, .171), (.044, .005, .014), orange)
        prop(car, "box", (.250, .105 * side, .090), (.020, .023, .004), ice,
             euler=f"0 .30 {-.20 * side}")
        prop(car, "box", (-.276, .087 * side, .078), (.004, .034, .007), "1 .065 .055 1")
        prop(car, "box", (-.012, .154 * side, .046), (.105, .003, .011), orange)

    for joint in ("front_left", "front_right", "rear_left", "rear_right"):
        wheel = next(body for body in car.iter("body") if body.find(f"joint[@name='{joint}']") is not None)
        side = 1 if joint.endswith("left") else -1
        prop(wheel, "cylinder", (0, .033 * side, 0), (.059, .002), ".31 .39 .44 1",
             euler="1.5707963 0 0")
        prop(wheel, "cylinder", (0, .036 * side, 0), (.047, .0015), graphite,
             euler="1.5707963 0 0")
        for angle in np.linspace(0., 2. * np.pi, 5, endpoint=False):
            prop(wheel, "box", (.027 * np.cos(angle), .038 * side, .027 * np.sin(angle)),
                 (.021, .0018, .004), ".69 .75 .78 1", euler=f"0 {-angle} 0")
        prop(wheel, "cylinder", (0, .041 * side, 0), (.014, .004), orange,
             euler="1.5707963 0 0")

    camera(car, "chase", [-1.33, -1.02, .84], [.50, .0, .11], 52)
    camera(car, "detail", [.85, -.80, .44], [.025, .0, .092], 41)


def dress_world(assets, world, track):
    ET.SubElement(assets, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1=".30 .49 .68", rgb2=".84 .90 .92", width="256", height="256")
    ET.SubElement(assets, "texture", name="turf_tex", type="2d", builtin="checker",
                  rgb1=".17 .36 .28", rgb2=".183 .377 .289", width="128", height="128")
    ET.SubElement(assets, "material", name="turf", texture="turf_tex", texrepeat="26 26",
                  texuniform="true", specular="0", shininess="0")
    floor = world.find("geom[@name='floor']")
    floor.set("material", "turf")
    floor.set("rgba", "1 1 1 1")
    floor.set("size", "64 64 .05")
    sun = world.find("light")
    sun.set("ambient", ".17 .18 .19")
    sun.set("diffuse", ".52 .50 .47")
    sun.set("specular", ".20 .19 .17")
    sun.set("castshadow", "false")
    ET.SubElement(world, "light", pos="-6 -8 9", dir=".42 .65 -1", directional="true",
                  castshadow="false", ambient=".31 .33 .36", diffuse=".29 .31 .33",
                  specular=".14 .16 .18")
    overview = world.find("camera[@name='overview']")
    world.remove(overview)
    camera(world, "overview", [12.8, -24.5, 32.], [1.3, -2.5, 0.], 43)

    tree_locations = [
        (-18., -9.5), (-17.6, -7.2), (-16.2, -5.4), (-13.4, -4.5), (-11., -4.),
        (-10., -1.), (-9., 2.4), (-7.7, 5.4), (-6.2, 8.7), (-3.1, 11.8),
        (0., 14.), (3.4, 14.6), (6.7, 13.2), (8.4, 10.6), (10.4, 8.7),
        (14.5, 8.2), (17., 6.), (17.7, 3.2), (17.2, .4), (18.2, -3.),
        (18.6, -6.5), (17.2, -10.), (14.6, -12.9), (10.5, -13.7),
        (6.2, -13.2), (-.4, -13.4), (-5., -13.4), (-9.4, -13.5), (-14.2, -13.1),
        (2.6, .6), (.65, -.8), (-2.4, -3.1), (8.4, -.65), (9.3, -2.5),
    ]
    for index, (x, y) in enumerate(tree_locations):
        if np.min(np.linalg.norm(track.points - [x, y], axis=1)) < 2.0:
            continue
        scale = .72 + .12 * (index % 4)
        prop(world, "ellipsoid", (x + .11, y - .045, .0015),
             (.32 * scale, .23 * scale, .002), ".025 .09 .07 .34")
        prop(world, "cylinder", (x, y, .28 * scale), (.065 * scale, .28 * scale), ".27 .19 .12 1")
        prop(world, "ellipsoid", (x, y, .68 * scale), (.31 * scale, .28 * scale, .40 * scale),
             ".075 .31 .22 1")
        prop(world, "ellipsoid", (x + .11, y, .93 * scale), (.24 * scale, .22 * scale, .32 * scale),
             ".12 .41 .28 1")

    # The pit lane and shelter are aligned with the start straight.
    center, idx = track.sample(0.)
    forward = track.tangents[idx]
    left = np.array([-forward[1], forward[0]])
    yaw = np.arctan2(forward[1], forward[0])
    rotation = f"0 0 {yaw}"

    def pit_box(along, offset, z, size, color):
        position = center + along * forward + offset * left
        prop(world, "box", (*position, z), size, color, euler=rotation)

    pit_box(.25, 1.91, .004, (4.05, .45, .004), ".26 .30 .31 1")
    pit_box(.25, 2.38, .012, (4.05, .024, .003), ".83 .79 .63 1")
    pit_box(.25, 3.08, .035, (3.65, .67, .035), ".30 .34 .35 1")
    pit_box(.25, 3.66, .34, (3.41, .062, .33), ".13 .22 .25 1")
    for along in (-2.95, -1.35, .25, 1.85, 3.45):
        pit_box(along, 2.61, .38, (.035, .035, .37), ".72 .77 .77 1")
    pit_box(.25, 3.14, .77, (3.51, .59, .045), ".085 .17 .21 1")
    pit_box(.25, 2.56, .77, (3.51, .02, .044), ".97 .33 .075 1")
