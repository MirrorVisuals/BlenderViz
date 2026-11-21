/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/light.h"
#include "scene/mesh.h"

#include "DNA_light_types.h"

#include "DNA_world_types.h"
#include "blender/sync.h"
#include "blender/util.h"
#include "scene/object.h"

CCL_NAMESPACE_BEGIN

void BlenderSync::sync_light(BObjectInfo &b_ob_info, Light *light)
{
  blender::Light &b_light = *blender::id_cast<blender::Light *>(b_ob_info.object_data);

  light->name = b_light.id.name + 2;

  if (PointLight *point_light = dynamic_cast<PointLight *>(light)) {
    point_light->set_radius(b_light.radius);
    point_light->set_is_sphere(!(b_light.mode & blender::LA_USE_SOFT_FALLOFF));
  }
  else if (AreaLight *area_light = dynamic_cast<AreaLight *>(light)) {
    area_light->set_sizeu(b_light.area_size);
    area_light->set_spread(b_light.area_spread);
    area_light->set_full_visible_surface((b_light.mode & blender::LA_FULL_VISIBLE_SURFACE) != 0);
    if (b_light.area_shape == blender::LA_AREA_SQUARE ||
        b_light.area_shape == blender::LA_AREA_DISK)
    {
      area_light->set_sizev(area_light->get_sizeu());
    }
    else {
      area_light->set_sizev(b_light.area_sizey);
    }
    area_light->set_ellipse(b_light.area_shape == blender::LA_AREA_DISK ||
                            b_light.area_shape == blender::LA_AREA_ELLIPSE);
  }
  else if (SunLight *sun_light = dynamic_cast<SunLight *>(light)) {
    sun_light->set_angle(b_light.sun_angle);
  }
  if (SpotLight *spot_light = dynamic_cast<SpotLight *>(light)) {
    spot_light->set_angle(b_light.spotsize);
    spot_light->set_smooth(b_light.spotblend);
  }

  blender::PointerRNA light_rna_ptr = RNA_id_pointer_create(&b_light.id);

  /* Color and strength. */
  float3 light_color = make_float3(b_light.r, b_light.g, b_light.b);
  if (b_light.mode & blender::LA_USE_TEMPERATURE) {
    float color[3];
    RNA_float_get_array(&light_rna_ptr, "temperature_color", color);
    light_color *= make_float3(color[0], color[1], color[2]);
  }

  const float3 strength = light_color * b_light.energy * exp2f(b_light.exposure);
  light->set_strength(strength);

  /* normalize */
  light->set_normalize(!(b_light.mode & blender::LA_UNNORMALIZED));

  /* shadow */
  blender::PointerRNA clight = RNA_pointer_get(&light_rna_ptr, "cycles");
  light->set_cast_shadow(b_light.mode & blender::LA_SHADOW);
  light->set_use_mis(get_boolean(clight, "use_multiple_importance_sampling"));

  /* caustics light */
  light->set_use_caustics(get_boolean(clight, "is_caustics_light"));

  light->set_max_bounces(get_int(clight, "max_bounces"));

  /* Attenuation by the ray length (Map Range between Min and Max Distance, Light Falloff). The RNA
   * enums list the types in the order of #LightMapRangeType and #LightFalloffType, without "none".
   */
  light->set_map_range_type(get_boolean(clight, "use_map_range") ?
                                get_enum(clight, "map_range_interpolation_type") + 1 :
                                0);
  light->set_min_distance(get_float(clight, "map_range_min_distance"));
  light->set_max_distance(get_float(clight, "map_range_max_distance"));
  light->set_map_range_to_min(get_float(clight, "map_range_to_min"));
  light->set_map_range_to_max(get_float(clight, "map_range_to_max"));
  light->set_map_range_steps(get_float(clight, "map_range_steps"));
  light->set_falloff_type(
      get_boolean(clight, "use_light_falloff") ? get_enum(clight, "light_falloff_output") + 1 : 0);
  light->set_falloff_smooth(get_float(clight, "light_falloff_smooth"));

  if (AreaLight *area_light = dynamic_cast<AreaLight *>(light)) {
    area_light->set_is_portal(get_boolean(clight, "is_portal"));
  }

  /* tag */
  light->tag_update(scene);
}

/* BlenderViz: Cylinder light. A tube along local Z, `area_size` long and `radius` thick, made as
 * an emissive mesh with the light's own shader (see #BlenderSync::sync_lights). */
void BlenderSync::sync_light_cylinder_mesh(BObjectInfo &b_ob_info, Mesh *mesh)
{
  const blender::Light &b_light = *blender::id_cast<blender::Light *>(b_ob_info.object_data);
  const float r = max(b_light.radius, 1e-5f);
  const float h = max(b_light.area_size, 1e-5f) * 0.5f;
  const int n = 48;

  /* A copy: setting a node array takes the data from the array it is given. */
  array<Node *> used_shaders = mesh->get_used_shaders();
  Mesh new_mesh;
  new_mesh.set_used_shaders(used_shaders);
  /* Side: two rings (smooth). Ends: two rings and two centers (flat). */
  const int num_verts = 4 * n + 2;
  const int num_tris = 2 * n + 2 * n;
  new_mesh.resize_mesh(num_verts, num_tris);
  packed_float3 *P = new_mesh.attributes.find(ATTR_STD_POSITION)->data_for_write<packed_float3>();
  for (int i = 0; i < n; i++) {
    const float a = M_2PI_F * float(i) / float(n);
    const float x = r * cosf(a), y = r * sinf(a);
    P[i] = make_float3(x, y, -h);
    P[n + i] = make_float3(x, y, h);
    P[2 * n + i] = make_float3(x, y, -h);
    P[3 * n + i] = make_float3(x, y, h);
  }
  const int bottom_center = 4 * n, top_center = 4 * n + 1;
  P[bottom_center] = make_float3(0.0f, 0.0f, -h);
  P[top_center] = make_float3(0.0f, 0.0f, h);

  int *tris = new_mesh.get_triangles().data();
  bool *smooth = new_mesh.get_smooth().data();
  int *shader = new_mesh.get_shader().data();
  int t = 0;
  auto add_tri = [&](const int a, const int b, const int c, const bool is_smooth) {
    tris[t * 3 + 0] = a;
    tris[t * 3 + 1] = b;
    tris[t * 3 + 2] = c;
    smooth[t] = is_smooth;
    shader[t] = 0;
    t++;
  };
  for (int i = 0; i < n; i++) {
    const int j = (i + 1) % n;
    /* Outward facing, counter-clockwise seen from outside. */
    add_tri(i, j, n + j, true);
    add_tri(i, n + j, n + i, true);
    add_tri(bottom_center, 2 * n + j, 2 * n + i, false);
    add_tri(top_center, 3 * n + i, 3 * n + j, false);
  }

  mesh->clear_non_sockets();
  for (const SocketType &socket : new_mesh.type->inputs) {
    if (socket.name == "use_motion_blur" || socket.name == "used_shaders") {
      continue;
    }
    mesh->set_value(socket, new_mesh, socket);
  }
  mesh->attributes.update(std::move(new_mesh.attributes));
  mesh->tag_update(scene, mesh->triangles_is_modified());
}

void BlenderSync::sync_background_light(blender::bScreen *b_screen, blender::View3D *b_v3d)
{
  blender::World *b_world = view_layer.world_override ? view_layer.world_override : b_scene->world;

  if (b_world) {
    blender::PointerRNA world_rna_ptr = RNA_id_pointer_create(&b_world->id);
    blender::PointerRNA cworld = RNA_pointer_get(&world_rna_ptr, "cycles");

    enum SamplingMethod { SAMPLING_NONE = 0, SAMPLING_AUTOMATIC, SAMPLING_MANUAL, SAMPLING_NUM };
    const int sampling_method = get_enum(
        cworld, "sampling_method", SAMPLING_NUM, SAMPLING_AUTOMATIC);
    const bool sample_as_light = (sampling_method != SAMPLING_NONE);

    /* Create object. */
    Object *object;
    const ObjectKey object_key(b_world, nullptr, b_world, false);
    bool update = object_map.add_or_update(&object, &b_world->id, &b_world->id, object_key);
    if (update) {
      /* Lights should be shadow catchers by default. */
      object->set_is_shadow_catcher(true);
    }

    object->set_lightgroup(
        ustring((b_world && b_world->lightgroup) ? b_world->lightgroup->name : ""));
    object->set_asset_name(ustring(b_world->id.name + 2));

    /* Create geometry. */
    const GeometryKey geom_key{b_world, Geometry::BACKGROUND_LIGHT};
    Geometry *geom = geometry_map.find(geom_key);
    if (geom) {
      update |= geometry_map.update(geom, &b_world->id);
    }
    else {
      geom = scene->create_light_node<BackgroundLight>();
      geometry_map.add(geom_key, geom);
      object->set_geometry(geom);
      update = true;
    }

    if (update || world_recalc || b_world != world_map) {
      /* Initialize light geometry. */
      BackgroundLight *light = static_cast<BackgroundLight *>(geom);

      array<Node *> used_shaders;
      used_shaders.push_back_slow(scene->default_background);
      light->set_used_shaders(used_shaders);

      if (sampling_method == SAMPLING_MANUAL) {
        light->set_map_resolution(get_int(cworld, "sample_map_resolution"));
      }
      else {
        light->set_map_resolution(0);
      }

      light->set_use_mis(sample_as_light);
      light->set_max_bounces(get_int(cworld, "max_bounces"));

      /* Caustic light. */
      light->set_use_caustics(get_boolean(cworld, "is_caustics_light"));

      light->set_cast_shadow(get_boolean(cworld, "use_shadows"));

      light->tag_update(scene);

      geometry_map.set_recalc(b_world);
    }
  }

  world_map = b_world;
  world_recalc = false;
  viewport_parameters = BlenderViewportParameters(b_screen, b_v3d, use_developer_ui);
}

CCL_NAMESPACE_END
