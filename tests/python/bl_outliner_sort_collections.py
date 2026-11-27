# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

# Outliner: Sort Collections (BlenderViz). With Sort Alphabetically, collections are sorted by
# name too (natural order). The sorting itself happens when the Outliner draws, which needs a
# window, so this checks the setting; the order is checked by eye in the Outliner.
# ./blender.bin --background --factory-startup --python tests/python/bl_outliner_sort_collections.py
import sys
import bpy

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


spaces = [space for screen in bpy.data.screens for area in screen.areas
          for space in area.spaces if space.type == 'OUTLINER']
check(len(spaces) > 0, "the startup file has Outliners")
space = spaces[0]
check(hasattr(space, "use_sort_collections"), "the Outliner has Sort Collections")
check(space.use_sort_collections, "on by default")
check(space.use_sort_alpha, "Sort Alphabetically on by default, as in Blender")
space.use_sort_collections = False
check(not space.use_sort_collections and space.use_sort_alpha,
      "can be turned off on its own, Sort Alphabetically stays on")
space.use_sort_collections = True
check(space.use_sort_collections, "and on again")

# The panel shows it under Sort Alphabetically.
import bl_ui.space_outliner as so
source = open(so.__file__, encoding="utf-8").read()
check('"use_sort_collections"' in source, "the filter popover shows it")

print()
if failures:
    print("{:d} FAILED:".format(len(failures)))
    for f in failures:
        print("  " + f)
    print("RESULT: FAILED")
    sys.exit(1)
print("RESULT: ALL PASSED")
