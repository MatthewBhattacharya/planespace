# PLANESPACE — Ruleset (v2.1, board-game edition)

## Overview
Two-player territory game on a 10×10 dot grid. Take turns placing polygons to score
points based on the **area** and **geometric type** of each shape. Scoring is pure
lookup: find two numbers on the tables below and add them — no square roots, no
fraction multiplication, no calculator needed.

## Setup
- 10×10 grid of dots (coordinates 0–9 on each axis)
- 7 **Gem Points** ★ at: (1,1), (8,1), (1,8), (8,8), (4,4), (6,3), (3,6)
- Player 1 = Blue, Player 2 = Red. Blue goes first.

## On Your Turn
1. Click 3–8 grid dots to choose your polygon's vertices
2. Click **Place Shape** to lock it in and score it
3. If you can't make a valid shape, click **Pass**
Two consecutive passes end the game.

## Placement Rules
- Polygon must be **simple** (no self-crossing edges)
- Area must be **> 0** (no collinear/degenerate shapes)
- Area must be **≤ 9 grid-unit²**
- Interior must **not overlap** any existing shape's interior
- You may share an edge or vertex freely with your **own** earlier shapes
- You may only share a **vertex** (not a full edge) with an **opponent's** shape —
  this is the main blocking tool: wall off territory by surrounding it without ever
  giving your opponent a flush edge to build against

## Scoring (immediate, per shape placed)

**Total = Base Score + Shape Bonus + Gem Bonus.** Compute area with the Shoelace
formula (it's always a multiple of 0.5 on this grid), then look up both tables below.

### Base Score Table (by area)
| Area  | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 4.5 | 5 | 5.5 | 6 | 6.5 | 7 | 7.5 | 8 | 8.5 | 9 |
|-------|-----|---|-----|---|-----|---|-----|---|-----|---|-----|---|-----|---|-----|---|-----|---|
| Score | 2   | 3 | 3   | 4 | 4   | 5 | 5   | 6 | 6   | 6 | 7   | 7 | 7   | 7 | 8   | 8 | 8   | 8 |

### Shape Bonus Table (by size bracket)
Small = area ≤ 3, Medium = area ≤ 6, Large = area ≤ 9.

| Shape (best tier wins)      | Small | Medium | Large |
|-----------------------------|-------|--------|-------|
| Any Triangle                | 10    | 8      | 5     |
| Isosceles Triangle          | 14    | 11     | 7     |
| Right Triangle               | 18    | 14     | 9     |
| 45-45-90 Triangle            | 22    | 17     | 11    |
| Equilateral Triangle ★ best  | 28    | 21     | 14    |
| Any Quadrilateral           | 6     | 5      | 3     |
| Trapezoid                   | 11    | 8      | 6     |
| Parallelogram               | 15    | 11     | 8     |
| Rectangle or Rhombus        | 20    | 15     | 10    |
| Square ★ best quad           | 30    | 23     | 15    |
| Pentagon (non-convex)       | 10    | 8      | 5     |
| Convex Pentagon             | 18    | 14     | 9     |
| Hexagon (non-convex)        | 13    | 10     | 7     |
| Convex Hexagon              | 24    | 18     | 12    |
| 7–8 sided (any)             | 8     | 6      | 4     |

### Gem Bonus
+10 per ★ Gem vertex used — **but each Gem only ever pays out once**, to whichever
shape (yours or your opponent's) touches it first. Later shapes touching the same
point score nothing extra from it, even if it's your own shape.

**Example**: a tiny 45-45-90 triangle (legs 2, area 2) that's first to touch a gem:
  Base (2) + Small 45-45-90 bonus (22) + first-touch gem (10) = **34 pts**.

## Strategy Notes
- Small, precise shapes still beat big sprawling ones: a 1×1 square scores **33 pts**
  (base 3 + Small bonus 30) versus a 3×3 square's **23 pts** (base 8 + Large bonus 15).
  Bigger area helps your Base Score a little, but it knocks you into a worse bonus
  bracket, which costs you more than you gained.
- The old corner-triangle "golden move" is geometrically impossible — three corner
  gems are 7 units apart, giving a triangle of area 24.5, far over the 9-unit cap.
- Three gems — (1,1), (4,4), (8,8) — are collinear on the diagonal y=x and cannot
  form a triangle together.
- Gems are a race: whoever's shape touches a gem vertex first banks it forever, so
  contesting a gem early denies your opponent that bonus for the rest of the game.
- Blocking is real: since you can't share a full edge with an opponent's shape, you
  can wall off a region by building a ring of your own shapes around it, denying
  your opponent any snug-fitting placement inside.

## Geometry Learned
- Area via Shoelace formula • Pythagorean theorem for right triangles
- 45-45-90 special triangle • Collinearity detection
- Quadrilateral hierarchy: Square ⊂ Rect ∩ Rhombus ⊂ Parallelogram ⊂ Trapezoid
- Convexity (same-sign cross products) • Pick's theorem (A = I + B/2 − 1)
- Diminishing returns at scale (size brackets) without ever computing a square root
