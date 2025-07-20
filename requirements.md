# 4D Rubik's Cube Application Requirements

This document outlines the requirements for the 4D Rubik's Cube application. It has been updated to reflect concrete decisions made during implementation.

## 1. Core Logic & Data Structures

*   **Cube Geometry & Visualization:** The 4D hypercube is represented as 8 separate 3x3x3 arrangements of individual colored 3D cubes.
    *   Each individual "sticker" of the 4D cube is visually represented as a small, single-colored 3D cube (0.8x scale with 1.2x spacing).
    *   There are 27 of these sticker-cubes per "side" of the 4D hypercube, arranged in a 3x3x3 grid.
    *   There are a total of 8 "sides" to the hypercube, positioned at distinct 4D coordinates.
    *   This results in a total of 8 * 27 = 216 visible sticker-cubes.
    *   **4D Positioning:** The 8 faces are positioned at the vertices of a tesseract:
        *   Face centers at coordinates with one axis fixed at ±1.0 (e.g., X=±1, Y=±1, Z=±1, W=±1)
        *   Each face is a 3x3x3 grid with stickers offset by ±2/3 units on the 3 free axes
        *   Face spacing parameter allows visual separation between faces
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
    *   **Compute Shader:** 4D transformations, projections, and face culling
    *   **Vertex/Fragment Shaders:** Final 3D rendering with visibility flags
    *   **Clear Shader:** Dark gray background for better visibility
    *   Per-instance colors and positions processed on GPU

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

*   **HypercubeApp Struct:** Main application state containing UI controls:
    *   Sticker scale and face scale parameters
    *   Current render mode (Standard, Normal Map, Depth Map)
    *   UI message handling and state updates
*   **HypercubeShaderProgram:** Custom iced shader widget managing 3D rendering:
    *   Camera controller and 4D rotation state
    *   Mouse and keyboard input handling
    *   Integration with iced's rendering pipeline
*   **Renderer Struct:** Pure GPU rendering system:
    *   wgpu resources (buffers, pipelines, textures)
    *   Instanced rendering for all 216 sticker cubes
    *   Multiple rendering modes with specialized shaders
*   **Clean Separation:** UI state, 3D logic, and graphics rendering are clearly separated with well-defined interfaces.

## 5. Technical Configuration

*   **Language:** Rust (2024 edition)
*   **UI Framework:** `iced` 0.13.1 with wgpu integration for modern GUI
*   **Graphics:** `wgpu` via iced for GPU-accelerated rendering
*   **Mathematics:** `nalgebra` 0.33.2 for 4D linear algebra
*   **Constants:**
    *   Sticker scale: 0.8 (size of individual cubes)
    *   Sticker spacing: 1.2 (gap between stickers)
    *   Mouse sensitivity: 0.5
    *   Projection FOV: 45°
    *   4D viewer distance: 3.0 units

## 6. Current Features (IMPLEMENTED)

*   **Complete 4D Hypercube Visualization:** All 8 faces properly positioned and rendered in 4D space
*   **Interactive 4D Exploration:** Users can rotate the hypercube in 4D space to see different cross-sections
*   **Proper 3D Navigation:** Standard 3D camera controls for viewing the projected hypercube from different angles
*   **4D Face Culling:** Culling shows only front-facing faces (typically 7 out of 8)
*   **Hardware Triangle Culling:** GPU-based backface culling
*   **Visual Depth Cues:** Depth testing provides visual ordering of sticker cubes
*   **Real-time Controls:** Responsive mouse controls with configurable sensitivity
*   **UI Framework:** iced-based interface with controls
*   **Interactive Controls:** Sliders for sticker scale and face spacing
*   **Debug Visualization:** Dropdown menu for switching between rendering modes
*   **Lighting System:** Directional lighting with normal calculations across all 4D transformations
*   **State Management:** Clean architecture supporting future game mechanics

## 7. Future Features (TO BE IMPLEMENTED)

### Game Mechanics
*   **Move Input:** Mechanism for selecting "slabs" of the hypercube to rotate
*   **4D Cube Mechanics:** Implementation of legal 4D Rubik's cube moves and rotations
*   **Scrambling:** Function to randomly scramble the cube state
*   **Move History:** Support for undoing and redoing moves
*   **UI Elements:** Controls for scrambling, move selection, and cube operations
*   **Solving Algorithms:** Advanced feature for automated solving
*   **Piece Classification:** Implementation of the 80 movable pieces (corners, edges, faces, cells)

### Visual Enhancements
*   **Lighting System:** Directional light sources with Phong lighting model provide enhanced depth perception ✓
*   **Shadow Rendering:** Add shadow mapping or shadow volumes to provide visual grounding for the hypercube

## 8. Performance Requirements (NON-FUNCTIONAL)

### Current Performance Status (IMPLEMENTED)
*   **Instanced Rendering:** ✓ Instanced rendering with 36 vertices per cube
*   **Vertex Shader Processing:** ✓ All 4D transformations moved to vertex shaders 
*   **Static Geometry:** ✓ Pre-generated cube vertices stored in vertex buffers
*   **GPU Resource Management:** ✓ Bind group layouts for different shader types
*   **CPU Normal Calculation:** ✓ Normals calculated on CPU, cached, and passed via uniforms

### Remaining Optimizations
*   **Branch Reduction:** Minimize conditional statements in shaders for better GPU performance
*   **Memory Layout:** Improve uniform buffer layouts and alignment

## 9. Technical Notes

*   **Performance:** Vertex shader-based instanced rendering processes all 216 cubes
*   **UI Integration:** Integration with iced framework for desktop application experience
*   **Culling System:** Dual-stage culling (4D face + hardware triangle) reduces overdraw
*   **Normal Calculation:** CPU-calculated normals provide lighting across 4D transformations
*   **Debug Features:** Multiple rendering modes (Standard/Normal/Depth) for development and debugging
*   **Extensibility:** Clean architecture designed to easily add game mechanics and move systems
*   **Cross-platform:** iced + wgpu support Windows, macOS, and Linux
*   **Memory Management:** GPU buffer management with vertex attributes and uniform buffers