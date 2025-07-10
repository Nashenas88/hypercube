# 4D Rubik's Cube Application Requirements

This document outlines the requirements for the 4D Rubik's Cube application. It will be updated as the design is refined.

## 1. Core Logic & Data Structures

*   **Cube Geometry & Visualization:** The 4D hypercube will be represented as an arrangement of 3D cubes.
    *   Each individual "sticker" of the 4D cube is visually represented as a small, single-colored 3D cube.
    *   There are 27 of these sticker-cubes per "side" of the 4D hypercube, arranged in a 3x3x3 grid.
    *   There are a total of 8 "sides" to the hypercube.
    *   This results in a total of 8 * 27 = 216 visible sticker-cubes.
*   **Pieces:** The 216 sticker-cubes belong to 80 movable pieces plus a central, non-visible core. The type of piece is determined by the number of stickers it has (1, 2, 3, or 4).
*   **Frame of Reference:** The center sticker-cube of each of the 8 "sides" has a fixed color and position relative to the other centers. This provides a stable frame for the puzzle. The 8 center colors define the puzzle's color scheme.
*   **State Management:** The application must track the state of the cube (the position and orientation of all pieces) after each move.
*   **Rotations:** The application must implement 4D rotations corresponding to twisting a "slab" of the hypercube.

## 2. Visualization & UI

*   **Projection:** The primary visualization will be the 8 separate 3x3x3 cubes, as described in the geometry section. This avoids complex 4D-to-2D projection logic for the main interaction.
*   **Rendering:** A custom Vulkan engine will be used for rendering the 3D cubes.
*   **UI Elements:** The UI will include controls for:
    *   Scrambling the cube.
    *   Undoing/redoing moves.
    *   Selecting slabs for rotation.
    *   Controlling the 3D camera.

## 3. User Interaction

*   **Move Input:** A clear mechanism must be designed for the user to select a "slab" of the hypercube to rotate.
*   **Camera Controls:** The user must be able to rotate the 3D view of the cube arrangements to see them from different angles.

## 4. Features

*   **Cube Size:** The initial version will be a 3x3x3x3 hypercube.
*   **Scrambling:** A function to randomly scramble the cube.
*   **Solving**: A solver would be a very advanced feature, but we can consider it.
*   **History:** Support for undoing and redoing moves.

## 5. Technology

*   **Language:** Rust
*   **Graphics:** A custom engine using a Vulkan wrapper.
