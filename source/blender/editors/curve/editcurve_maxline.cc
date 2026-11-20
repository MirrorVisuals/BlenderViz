/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edcurve
 *
 * BlenderViz: Max Line, draw a spline directly, as the 3ds Max Line tool.
 *
 * Points are placed with the same snapping cursor as the Add tools (the grid on the surface or
 * the placement plane, the scene's snap settings, Ctrl to toggle snapping), so placing a point
 * costs no more than moving the Add tool. Segments are straight. The finished spline is turned
 * into a curve object by the Python operator `curve.maxline_commit` (`viz_maxline.py`), which
 * applies the Max Line settings (origin, alignment to the surface object, parenting...).
 */

#include <string>

#include "MEM_guardedalloc.h"

#include "BLI_kdopbvh.hh"
#include "BLI_math_vector.h"
#include "BLI_math_vector_types.hh"
#include "BLI_vector.hh"

#include "BLT_translation.hh"

#include "BKE_context.hh"
#include "BKE_report.hh"
#include "BKE_screen.hh"

#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_query.hh"

#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "DNA_screen_types.h"
#include "DNA_userdef_types.h"

#include "WM_api.hh"
#include "WM_types.hh"

#include "ED_screen.hh"
#include "ED_space_api.hh"
#include "ED_transform_snap_object_context.hh"
#include "ED_view3d.hh"

#include "GPU_immediate.hh"
#include "GPU_state.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "UI_interface_icons.hh"
#include "UI_resources.hh"

#include "curve_intern.hh"

namespace blender {

/** Clicking within this many pixels of the first point closes the spline. */
#define MAXLINE_CLOSE_RADIUS_PX 12.0f

struct MaxLinePoint {
  float3 co;
  /** Name of the object the point was placed on, empty in empty space. */
  std::string object;
  float3 normal;
  bool has_normal;
};

struct MaxLineData {
  ARegion *region;
  void *draw_handle;
  V3DSnapCursorState *snap_state;

  Vector<MaxLinePoint> points;
  /** Where the next point goes (the snapping cursor). */
  float3 cursor;
  /** The mouse is over the first point: a click closes the spline. */
  bool hover_first;
  bool from_tool;
};

static void maxline_cursor_update(bContext *C, MaxLineData *mld, const int2 &mval)
{
  ED_view3d_cursor_snap_data_update(mld->snap_state, C, mld->region, mval);
  const V3DSnapCursorData *snap_data = ED_view3d_cursor_snap_data_get();
  mld->cursor = float3(snap_data->loc);

  mld->hover_first = false;
  if (mld->points.size() >= 3) {
    float2 first_px;
    if (ED_view3d_project_float_global(
            mld->region, mld->points.first().co, first_px, V3D_PROJ_TEST_NOP) == V3D_PROJ_RET_OK)
    {
      const float radius = MAXLINE_CLOSE_RADIUS_PX * UI_SCALE_FAC;
      mld->hover_first = len_squared_v2v2(first_px, float2(mval)) <= radius * radius;
    }
  }
}

/** The object under the mouse and its surface normal, to place the spline in its space. */
static void maxline_point_surface(bContext *C,
                                  MaxLineData *mld,
                                  const int2 &mval,
                                  MaxLinePoint &r_point)
{
  r_point.object.clear();
  r_point.has_normal = false;

  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  View3D *v3d = CTX_wm_view3d(C);
  Scene *scene = CTX_data_scene(C);
  float ray_start[3], ray_normal[3];
  if (!ED_view3d_win_to_ray_clipped(
          depsgraph, mld->region, v3d, float2(mval), ray_start, ray_normal, true))
  {
    return;
  }

  ed::transform::SnapObjectContext *sctx = ED_view3d_cursor_snap_context_ensure(scene);
  ed::transform::SnapObjectParams params{};
  params.snap_target_select = SCE_SNAP_TARGET_ALL;
  params.edit_mode_type = ed::transform::SNAP_GEOM_FINAL;
  params.occlusion_test = ed::transform::SNAP_OCCLUSION_AS_SEEM;

  float ray_depth = BVH_RAYCAST_DIST_MAX;
  float loc[3], nor[3], obmat[4][4];
  int index;
  const Object *ob_eval = nullptr;
  if (ed::transform::snap_object_project_ray_ex(sctx,
                                                depsgraph,
                                                v3d,
                                                &params,
                                                ray_start,
                                                ray_normal,
                                                &ray_depth,
                                                loc,
                                                nor,
                                                &index,
                                                &ob_eval,
                                                obmat) &&
      ob_eval)
  {
    const Object *ob = DEG_get_original(ob_eval);
    r_point.object = ob->id.name + 2;
    if (normalize_v3_v3(r_point.normal, nor) > 0.0f) {
      r_point.has_normal = true;
    }
  }
}

static void maxline_draw(const bContext * /*C*/, ARegion *region, void *arg)
{
  MaxLineData *mld = static_cast<MaxLineData *>(arg);
  if (region != mld->region || mld->points.is_empty()) {
    return;
  }

  Vector<float3> line;
  for (const MaxLinePoint &point : mld->points) {
    line.append(point.co);
  }
  line.append(mld->hover_first ? mld->points.first().co : mld->cursor);

  GPU_blend(GPU_BLEND_ALPHA);
  GPU_depth_test(GPU_DEPTH_NONE);

  float viewport[4];
  GPU_viewport_size_get_f(viewport);

  GPUVertFormat *format = immVertexFormat();
  const uint pos = GPU_vertformat_attr_add(format, "pos", gpu::VertAttrType::SFLOAT_32_32_32);

  /* The spline: straight segments, anti-aliased. */
  immBindBuiltinProgram(GPU_SHADER_3D_POLYLINE_UNIFORM_COLOR);
  immUniform2fv("viewportSize", &viewport[2]);
  immUniform1f("lineWidth", 2.0f * U.pixelsize);
  immUniform1i("lineSmooth", 1);
  if (line.size() >= 3) {
    immUniformColor4f(0.10f, 0.85f, 1.0f, 1.0f);
    immBegin(GPU_PRIM_LINE_STRIP, line.size() - 1);
    for (const int i : line.index_range().drop_back(1)) {
      immVertex3fv(pos, line[i]);
    }
    immEnd();
  }
  /* The segment still being drawn. */
  immUniformColor4f(1.0f, 0.70f, 0.15f, 0.9f);
  immBegin(GPU_PRIM_LINES, 2);
  immVertex3fv(pos, line[line.size() - 2]);
  immVertex3fv(pos, line.last());
  immEnd();
  immUnbindProgram();

  /* The points, the first one marked: clicking it closes the spline. */
  GPU_program_point_size(true);
  immBindBuiltinProgram(GPU_SHADER_3D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_AA);
  immUniform1f("size", 7.0f * U.pixelsize);
  if (mld->points.size() >= 2) {
    immUniformColor4f(0.95f, 0.95f, 0.95f, 1.0f);
    immBegin(GPU_PRIM_POINTS, mld->points.size() - 1);
    for (const int i : mld->points.index_range().drop_front(1)) {
      immVertex3fv(pos, mld->points[i].co);
    }
    immEnd();
  }
  immUniform1f("size", (mld->hover_first ? 11.0f : 8.0f) * U.pixelsize);
  immUniformColor4f(1.0f, 0.35f, 0.35f, 1.0f);
  immBegin(GPU_PRIM_POINTS, 1);
  immVertex3fv(pos, mld->points.first().co);
  immEnd();
  immUnbindProgram();
  GPU_program_point_size(false);

  GPU_blend(GPU_BLEND_NONE);
}

static void maxline_status(bContext *C, const MaxLineData *mld)
{
  WorkspaceStatus status(C);
  status.item(IFACE_("Point"), ICON_MOUSE_LMB);
  status.item(IFACE_("Snap Toggle"), ICON_EVENT_CTRL);
  if (!mld->points.is_empty()) {
    status.item(IFACE_("Remove Last Point"), ICON_EVENT_BACKSPACE);
  }
  if (mld->points.size() >= 3) {
    status.item(IFACE_("Close"), ICON_EVENT_C);
  }
  status.item(IFACE_("Finish"), ICON_MOUSE_RMB, ICON_EVENT_RETURN);
  status.item(IFACE_("Cancel"), ICON_EVENT_ESC);
}

static void maxline_exit(bContext *C, wmOperator *op)
{
  MaxLineData *mld = static_cast<MaxLineData *>(op->customdata);
  if (mld == nullptr) {
    return;
  }
  ED_region_draw_cb_exit(mld->region->runtime->type, mld->draw_handle);
  ED_view3d_cursor_snap_state_free(mld->snap_state);
  ED_workspace_status_text(C, nullptr);
  ED_region_tag_redraw(mld->region);
  MEM_delete(mld);
  op->customdata = nullptr;
}

/** Make the curve object, with the Max Line settings, in Python. */
static bool maxline_commit(bContext *C, wmOperator *op, MaxLineData *mld, const bool cyclic)
{
  if (mld->points.size() < 2) {
    return false;
  }
  wmOperatorType *ot = WM_operatortype_find("CURVE_OT_maxline_commit", true);
  if (ot == nullptr) {
    BKE_report(op->reports, RPT_ERROR, "Max Line is not available (viz_maxline.py not loaded)");
    return false;
  }
  PointerRNA ptr = WM_operator_properties_create_ptr(ot);
  for (const MaxLinePoint &point : mld->points) {
    PointerRNA item;
    RNA_collection_add(&ptr, "points", &item);
    RNA_float_set_array(&item, "co", point.co);
    RNA_string_set(&item, "object", point.object.c_str());
    RNA_float_set_array(&item, "normal", point.normal);
    RNA_boolean_set(&item, "has_normal", point.has_normal);
  }
  RNA_boolean_set(&ptr, "cyclic", cyclic);
  const wmOperatorStatus status = WM_operator_name_call_ptr(
      C, ot, wm::OpCallContext::ExecDefault, &ptr, nullptr);
  WM_operator_properties_free(&ptr);
  return status & OPERATOR_FINISHED;
}

static bool maxline_continuous(bContext *C)
{
  Scene *scene = CTX_data_scene(C);
  PointerRNA scene_ptr = RNA_id_pointer_create(&scene->id);
  PropertyRNA *prop = RNA_struct_find_property(&scene_ptr, "maxline_settings");
  if (prop == nullptr) {
    return false;
  }
  PointerRNA settings = RNA_property_pointer_get(&scene_ptr, prop);
  return settings.data && RNA_boolean_get(&settings, "continuous");
}

static wmOperatorStatus maxline_finish(bContext *C, wmOperator *op, const bool cyclic)
{
  MaxLineData *mld = static_cast<MaxLineData *>(op->customdata);
  const bool created = maxline_commit(C, op, mld, cyclic);
  mld->points.clear();
  mld->hover_first = false;
  ED_view3d_cursor_snap_state_prevpoint_set(mld->snap_state, nullptr);
  ED_region_tag_redraw(mld->region);

  /* With the tool, the next click starts the next line. From the menu, Keep Tool Active. */
  if (!mld->from_tool && maxline_continuous(C)) {
    maxline_status(C, mld);
    return OPERATOR_RUNNING_MODAL;
  }
  maxline_exit(C, op);
  return created ? OPERATOR_FINISHED : OPERATOR_CANCELLED;
}

static void maxline_add_point(bContext *C, MaxLineData *mld, const int2 &mval)
{
  MaxLinePoint point;
  point.co = mld->cursor;
  maxline_point_surface(C, mld, mval, point);
  mld->points.append(std::move(point));
  /* Snapping relative to the last point (Perpendicular) and its line to the cursor. */
  ED_view3d_cursor_snap_state_prevpoint_set(mld->snap_state, mld->points.last().co);
  ED_region_tag_redraw(mld->region);
}

static bool maxline_is_navigation(const wmEvent *event)
{
  const int type = event->type;
  return ISMOUSE_WHEEL(type) || ISMOUSE_GESTURE(type) || ISNDOF(type) || type == MIDDLEMOUSE ||
         (type >= EVT_PAD0 && type <= EVT_PAD9) ||
         ELEM(type, EVT_PADPERIOD, EVT_PADSLASHKEY, EVT_PADMINUS, EVT_PADPLUSKEY, EVT_HOMEKEY);
}

static wmOperatorStatus maxline_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  MaxLineData *mld = static_cast<MaxLineData *>(op->customdata);

  if (maxline_is_navigation(event)) {
    return OPERATOR_PASS_THROUGH;
  }

  const int2 mval = int2(event->xy) - int2(mld->region->winrct.xmin, mld->region->winrct.ymin);
  const bool inside = mval.x >= 0 && mval.y >= 0 && mval.x < mld->region->winx &&
                      mval.y < mld->region->winy;

  if (ISMOUSE_MOTION(event->type) || ELEM(event->type, EVT_LEFTCTRLKEY, EVT_RIGHTCTRLKEY)) {
    if (inside) {
      maxline_cursor_update(C, mld, mval);
      ED_region_tag_redraw(mld->region);
    }
    return OPERATOR_RUNNING_MODAL;
  }

  if (event->val != KM_PRESS) {
    return OPERATOR_RUNNING_MODAL;
  }

  switch (event->type) {
    case LEFTMOUSE: {
      if (!inside) {
        break;
      }
      maxline_cursor_update(C, mld, mval);
      if (mld->hover_first) {
        return maxline_finish(C, op, true);
      }
      maxline_add_point(C, mld, mval);
      maxline_status(C, mld);
      break;
    }
    case EVT_BACKSPACEKEY:
    case EVT_DELKEY: {
      if (!mld->points.is_empty()) {
        mld->points.remove_last();
        ED_view3d_cursor_snap_state_prevpoint_set(
            mld->snap_state, mld->points.is_empty() ? nullptr : &mld->points.last().co.x);
        maxline_cursor_update(C, mld, mval);
        maxline_status(C, mld);
        ED_region_tag_redraw(mld->region);
      }
      break;
    }
    case EVT_CKEY: {
      if (mld->points.size() >= 3) {
        return maxline_finish(C, op, true);
      }
      break;
    }
    case RIGHTMOUSE:
    case EVT_RETKEY:
    case EVT_PADENTER:
    case EVT_SPACEKEY: {
      if (mld->points.size() >= 2) {
        return maxline_finish(C, op, false);
      }
      maxline_exit(C, op);
      return OPERATOR_CANCELLED;
    }
    case EVT_ESCKEY: {
      if (!mld->points.is_empty()) {
        mld->points.clear();
        ED_view3d_cursor_snap_state_prevpoint_set(mld->snap_state, nullptr);
        maxline_status(C, mld);
        ED_region_tag_redraw(mld->region);
        break;
      }
      maxline_exit(C, op);
      return OPERATOR_CANCELLED;
    }
    default:
      break;
  }
  return OPERATOR_RUNNING_MODAL;
}

static wmOperatorStatus maxline_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  ARegion *region = CTX_wm_region(C);
  if (region == nullptr || region->regiontype != RGN_TYPE_WINDOW) {
    /* From the sidebar or a menu of the 3D viewport: draw in its main region. */
    ScrArea *area = CTX_wm_area(C);
    region = area ? BKE_area_find_region_type(area, RGN_TYPE_WINDOW) : nullptr;
  }
  if (region == nullptr || region->regiondata == nullptr || CTX_wm_view3d(C) == nullptr) {
    BKE_report(op->reports, RPT_WARNING, "Max Line needs a 3D viewport");
    return OPERATOR_CANCELLED;
  }

  MaxLineData *mld = MEM_new<MaxLineData>(__func__);
  mld->region = region;
  mld->from_tool = RNA_boolean_get(op->ptr, "from_tool");
  mld->snap_state = ED_view3d_cursor_snap_state_create();
  mld->snap_state->draw_point = true;
  mld->snap_state->draw_plane = true;
  mld->draw_handle = ED_region_draw_cb_activate(
      region->runtime->type, maxline_draw, mld, REGION_DRAW_POST_VIEW);
  op->customdata = mld;

  const int2 mval = int2(event->xy) - int2(region->winrct.xmin, region->winrct.ymin);
  maxline_cursor_update(C, mld, mval);
  if (mld->from_tool && event->type == LEFTMOUSE && event->val == KM_PRESS) {
    /* A click with the Max Line tool: that click is the first point. */
    maxline_add_point(C, mld, mval);
  }

  maxline_status(C, mld);
  WM_event_add_modal_handler(C, op);
  ED_region_tag_redraw(region);
  return OPERATOR_RUNNING_MODAL;
}

static void maxline_cancel(bContext *C, wmOperator *op)
{
  maxline_exit(C, op);
}

static bool maxline_poll(bContext *C)
{
  if (CTX_data_mode_enum(C) != CTX_MODE_OBJECT) {
    CTX_wm_operator_poll_msg_set(C, "Only available in Object Mode");
    return false;
  }
  return ED_operator_view3d_active(C);
}

void CURVE_OT_maxline_draw(wmOperatorType *ot)
{
  ot->name = "Max Line";
  ot->idname = "CURVE_OT_maxline_draw";
  ot->description =
      "Draw a spline directly, as the 3ds Max Line tool: click to place points (straight "
      "segments) with the snapping cursor of the Add tools, Ctrl to toggle snapping, the first "
      "point or C to close, right-click or Enter to finish";

  ot->invoke = maxline_invoke;
  ot->modal = maxline_modal;
  ot->cancel = maxline_cancel;
  ot->poll = maxline_poll;

  ot->flag = OPTYPE_BLOCKING;

  PropertyRNA *prop = RNA_def_boolean(
      ot->srna,
      "from_tool",
      false,
      "From Tool",
      "Started by a click with the Max Line tool: the click places the first point");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);
}

}  // namespace blender
