# math.rs

CPU-side 4D rotation matrices for the 6 rotation planes, generic `create_4d_plane_rotation`, 4D→3D perspective projection (`project_cube_point`). Also `decompose_so4`/`compose_so4`, an isoclinic (biquaternion) decomposition of an arbitrary `SO(4)` rotation matrix into a pair of unit quaternions - used to animate the 4D orientation back to identity via quaternion slerp, since a single plane rotation (`shortest_arc_plane`) can only align one vector, not undo a whole accumulated orientation.
