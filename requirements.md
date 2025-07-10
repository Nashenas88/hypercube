# 4D Rubik's Cube Application Requirements

This document outlines the requirements for the 4D Rubik's Cube application. It has been updated to reflect concrete decisions made during implementation.

## 1. Core Logic & Data Structures

*   **Cube Geometry & Visualization:** The 4D hypercube is represented as 8 separate 3x3x3 arrangements of individual colored 3D cubes.
    *   Each individual "sticker" of the 4D cube is visually represented as a small, single-colored 3D cube (0.8x scale with 1.2x spacing).
    *   There are 27 of these sticker-cubes per "side" of the 4D hypercube, arranged in a 3x3x3 grid.
    *   There are a total of 8 "sides" to the hypercube, positioned at distinct 4D coordinates.
    *   This results in a total of 8 * 27 = 216 visible sticker-cubes.
    *   **4D Positioning:** The 8 sides are positioned as follows:
        *   7 sides arranged around a center (0,0,0) in the W=-2.0 plane, offset by ±4 units along X, Y, and Z axes
        *   1 side at the center (0,0,0) of the W=+2.0 plane
*   **Color Scheme:** 8 distinct colors define the puzzle: White, Yellow, Blue, Green, Red, Orange, Purple, Brown.
*   **Frame of Reference:** The center sticker-cube of each of the 8 "sides" has a fixed color and position relative to the other centers. This provides a stable frame for the puzzle.
*   **State Management:** The application tracks the state of the cube through the `Hypercube` struct containing `Side` structs with individual `Sticker` positions and colors.
*   **4D Mathematics:** Uses `nalgebra` for 4D vector operations and 4x4 matrix transformations.

## 2. Visualization & Rendering (IMPLEMENTED)

*   **Graphics Technology:** `wgpu` is used instead of Vulkan for better safety, portability, and ease of use.
*   **4D Projection:** Implements perspective projection from 4D to 3D space with configurable viewer distance (3.0 units).
*   **Rendering Pipeline:**
    *   Instanced rendering for all 216 sticker cubes
    *   Depth testing for proper z-ordering
    *   Per-instance model matrices and colors
    *   Real-time instance buffer updates for smooth 4D transformations
*   **Shader System:** Custom WGSL shaders supporting:
    *   Per-vertex position data
    *   Per-instance model matrices (4x4) and colors (RGBA)
    *   Camera view-projection matrices via uniform buffers

## 3. User Interaction (IMPLEMENTED)

*   **Camera Controls:** 3D orbital camera system around the hypercube origin:
    *   Right mouse drag: 3D camera rotation (yaw/pitch with -89° to +89° pitch clamping)
    *   Mouse wheel: Zoom in/out (5.0 to 50.0 units distance)
    *   Automatic aspect ratio adjustment on window resize
*   **4D Rotation Controls:** Interactive 4D hypercube rotation:
    *   Right mouse drag + Shift: 4D rotation in XW and YW planes
    *   Accumulative rotation matrix for complex 4D transformations
    *   Real-time visual feedback of 4D rotations
*   **Input System:** Clean separation of input handling between App state and rendering system.

## 4. Architecture (IMPLEMENTED)

*   **App Struct:** Contains all application state:
    *   Hypercube data structure
    *   Camera and projection settings
    *   4D rotation matrix
    *   Input state tracking
*   **Renderer Struct:** Pure rendering system that accepts external data:
    *   Graphics resources (buffers, pipelines, textures)
    *   Methods for updating instance data and rendering frames
*   **Clean Separation:** App handles logic and input, Renderer handles graphics, with clear interfaces.

## 5. Technical Configuration

*   **Language:** Rust (2024 edition)
*   **Graphics:** `wgpu` 0.20 with `winit` 0.29 for windowing
*   **Mathematics:** `nalgebra` 0.32 for 4D linear algebra
*   **Constants:**
    *   Sticker scale: 0.8 (size of individual cubes)
    *   Sticker spacing: 1.2 (gap between stickers)
    *   Mouse sensitivity: 0.5
    *   Projection FOV: 45°
    *   4D viewer distance: 3.0 units

## 6. Current Features (IMPLEMENTED)

*   **Complete 4D Hypercube Visualization:** All 8 sides properly positioned and rendered in 4D space
*   **Interactive 4D Exploration:** Users can rotate the hypercube in 4D space to see different cross-sections
*   **Proper 3D Navigation:** Standard 3D camera controls for viewing the projected hypercube from different angles
*   **Visual Depth Cues:** Depth testing ensures correct visual ordering of sticker cubes
*   **Smooth Real-time Controls:** Responsive mouse controls with configurable sensitivity
*   **State Management:** Clean architecture supporting future game mechanics

## 7. Future Features (TO BE IMPLEMENTED)

*   **Move Input:** Mechanism for selecting "slabs" of the hypercube to rotate
*   **4D Cube Mechanics:** Implementation of legal 4D Rubik's cube moves and rotations
*   **Scrambling:** Function to randomly scramble the cube state
*   **Move History:** Support for undoing and redoing moves
*   **UI Elements:** Controls for scrambling, move selection, and cube operations
*   **Solving Algorithms:** Advanced feature for automated solving
*   **Piece Classification:** Implementation of the 80 movable pieces (corners, edges, faces, cells)

## 8. Technical Notes

*   **Performance:** Instanced rendering supports smooth real-time interaction with 216 individual cubes
*   **Extensibility:** Architecture designed to easily add game mechanics and move systems
*   **Cross-platform:** `wgpu` ensures compatibility across Windows, macOS, and Linux
*   **Memory Management:** Efficient GPU buffer management with proper usage flags for dynamic updates