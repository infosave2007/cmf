# Prompt: Interactive 3D Glass Aquarium with Clownfish

## Task

Create a single-file HTML page featuring a realistic interactive 3D glass aquarium with three **clownfish** (Amphiprion ocellaris) swimming inside. Use **Three.js** (import from CDN) for all 3D rendering. Everything — geometry, materials, animation, lighting, controls — must be contained in one self-sufficient `.html` file with no external assets (no image textures, no model files). All fish, plants, stones, and the aquarium itself must be built procedurally using Three.js geometry and shaders.

---

## Scene Composition

### Background
- Deep ocean-blue gradient background (`#0a1628` → `#1a3a5c`) filling the entire viewport behind the aquarium.
- Optional: subtle animated caustic light pattern on the background to simulate underwater ambient light.

### Glass Aquarium (rectangular tank)
- Dimensions approximately 6 × 3 × 4 (width × depth × height) world units.
- **Glass material**: use `MeshPhysicalMaterial` with `transmission: 0.95`, `roughness: 0.05`, `thickness: 0.15`, `ior: 1.5`, `transparent: true`, `opacity: 0.18`, slight green-blue tint (`#e0f7fa`). The aquarium must look like real glass — you should see reflections, refractions, and the objects inside through it.
- Render the tank as 5 separate glass panels (4 walls + bottom). The top is open.
- Add a subtle dark wooden or matte-black frame/rim along the top edge of the tank for realism.

### Water
- A slightly transparent volume inside the tank (a box slightly smaller than the glass walls).
- Material: semi-transparent blue (`#1e88e5`, opacity ~0.18), with a subtle animated surface on top simulating gentle water ripples (use vertex displacement on the top face via a custom shader or sine-wave animation).

---

## Clownfish (×3)

Each fish must be a detailed **multi-part 3D model built from Three.js geometries**, NOT a flat sprite or a simple sphere. The fish species is **clownfish** (orange body with white vertical stripes bordered by thin black edges).

### Fish Anatomy — Required Parts

| Part | Geometry approach | Details |
|---|---|---|
| **Body** | Elongated ellipsoid (`SphereGeometry` scaled ~1.6× on Z, ~0.55× on X) or a custom `LatheGeometry` / `TubeGeometry` profile curve for a more organic fish silhouette | Orange (`#FF6D00`) base with 3 white vertical stripes (use shader or multiple merged geometries) bordered by thin black lines |
| **Tail (caudal fin)** | Flat fan shape via `ShapeGeometry` or `ExtrudeGeometry` with a forked or rounded profile | Must be rigged for **continuous left-right wagging animation** (oscillate rotation around Y-axis, amplitude ~±20°, synced with swimming speed) |
| **Dorsal fin** | Thin triangular or sail-like shape on top of the body | Slight procedural sway animation |
| **Pectoral fins (×2)** | Small triangular shapes on each side | Subtle flapping animation |
| **Anal fin** | Small triangular shape on the underside near the tail | Static or slight sway |
| **Eyes (×2)** | Sphere with black pupil (smaller dark sphere) and a white/glossy specular highlight dot | Placed on each side of the head; should be clearly visible and give the fish a "lively" look |
| **Mouth** | Small indentation or dark circle at the front | Optional but adds realism |

### Fish Coloring
- Body: bright orange `#FF6D00`
- Three vertical white stripes (head, mid-body, near tail) each bordered by a thin black edge
- Fins and tail: orange with white edging or translucent orange
- Belly: slightly lighter orange

### Fish Behavior
- Each fish swims **autonomously inside the aquarium bounds** with smooth, natural-looking movement:
  - Random waypoint navigation: pick a random target point inside the tank, smoothly turn and swim toward it, then pick a new target.
  - Fish should **never clip through the walls** — keep a margin of ~0.3 units from all tank boundaries.
  - Smooth rotation: the fish body should gradually rotate (slerp) to face the direction of movement. Fish bank slightly when turning (tilt on Z-axis).
  - Varying speed: each fish should have a slightly different base speed (0.005–0.015 units/frame) to look natural.
- **Tail wagging** is continuous but its frequency increases with swimming speed.
- The three fish should have slightly different sizes (scale 0.9, 1.0, 1.1) for variety.

---

## Aquarium Interior Details

### Bubbles
- A stream of bubbles rising from one or two points on the bottom of the tank.
- Each bubble: small translucent sphere (`MeshPhysicalMaterial`, transmission, slight iridescence if possible).
- Bubbles rise with slight horizontal wobble (sine wave on X/Z), shrink slightly as they approach the surface, and disappear at the waterline.
- Spawn rate: ~2–4 bubbles per second per stream.

### Stones / Gravel Floor
- Scatter 15–25 rounded stones across the bottom of the tank.
- Use `SphereGeometry` or `DodecahedronGeometry` with slight random scale, squash (flatten Y), and color variation (grays, browns, tans: `#8D6E63`, `#A1887F`, `#9E9E9E`, `#6D4C41`).
- Some stones should be larger, some tiny, creating a natural gravel bed.

### Aquatic Plants (×3–5 clusters)
- Each plant is a cluster of 4–8 tall, narrow blade-like shapes (`PlaneGeometry` or `ShapeGeometry`, double-sided material).
- Color: rich greens with variation (`#2E7D32`, `#388E3C`, `#1B5E20`, `#4CAF50`).
- **Sway animation**: each blade gently oscillates using a vertex shader or sine-based rotation at the base, with phase offsets per blade so they don't move in unison. The motion should look like a gentle underwater current.
- Place plant clusters at the back and corners of the tank.

---

## Lighting

- **Ambient light**: soft blue-white (`#b3e5fc`, intensity 0.4) for overall fill.
- **Directional light** (simulating room light from above-front): white-warm (`#fff8e1`, intensity 1.0), casting soft shadows.
- **Point light** inside the tank (simulating an aquarium lamp): positioned at the top-center of the tank, warm white (`#ffe0b2`, intensity 0.8), with distance falloff.
- Optional: a subtle volumetric/god-ray effect or caustic pattern projected onto the tank bottom.

---

## Camera & Controls

- Use `OrbitControls` from Three.js addons (import from CDN: `three/addons/controls/OrbitControls.js`).
- **Default camera position**: slightly above and in front of the aquarium, angled to show a nice 3/4 perspective.
- The user can:
  - **Rotate** the scene (left-click drag) — orbit around the aquarium.
  - **Zoom** (scroll wheel) — get close to see fish details or zoom out to see the full scene.
  - **Pan** (right-click drag) — shift the view.
- Set `minDistance` and `maxDistance` on controls to prevent zooming too far in/out.
- Set the orbit target to the center of the aquarium.

---

## Animation & Performance

- Use `requestAnimationFrame` loop for smooth 60fps rendering.
- Animate per frame:
  - Fish movement + tail/fin wagging
  - Bubble rising + wobble
  - Plant blade swaying
  - Water surface ripple
- Use `renderer.setPixelRatio(window.devicePixelRatio)` and handle `window.resize` events.
- Enable `renderer.shadowMap` for soft shadows.
- Use `toneMapping: THREE.ACESFilmicToneMapping` and `toneMappingExposure: 1.2` for cinematic look.

---

## Technical Requirements

- **Single HTML file**, no external dependencies except CDN imports.
- Import Three.js and OrbitControls via ES module imports from `https://cdnjs.cloudflare.com/ajax/libs/three.js/` or `https://unpkg.com/three@0.160.0/build/three.module.js` (and corresponding addons path).
- All code in `<script type="module">`.
- Responsive: fill the full browser viewport (`width: 100vw; height: 100vh; margin: 0; overflow: hidden`).
- No console errors, no missing textures, no broken imports.

---

## Quality Bar

The final result should look like a **polished interactive 3D demo** — something impressive enough to screenshot and share. The clownfish should be clearly recognizable as fish (not blobs), with visible eyes, moving tails, and distinct orange-white-black coloring. The glass aquarium should have a realistic transparent look. The overall scene should feel alive and calming, like watching a real aquarium.
