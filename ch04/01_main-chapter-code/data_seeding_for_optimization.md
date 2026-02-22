# 🌱 Data Seeding for Optimization

Simple explanations of the **vertical data** and **spiral data** generators used when learning optimization strategies (e.g. in the *Optimization_intro_neural_networks* notebook). Think of them as two kinds of “practice puzzles” we give the neural network: one easy (vertical strips), one tricky (spirals).

---

## 📊 Why Two Datasets?

| Dataset        | Difficulty | Looks like              | Good for                                      |
|----------------|------------|-------------------------|-----------------------------------------------|
| **Vertical**   | 🟢 Easy    | Three vertical bands    | Seeing that a strategy can work at all        |
| **Spiral**     | 🔴 Hard    | Three winding arms      | Seeing that naive strategies aren’t enough   |

We use **vertical** to show that “randomly adjust and keep improvements” can reduce loss and raise accuracy. We use **spiral** to show that the same idea fails on harder data, so we need **gradient-based** optimization (calculus, backprop).

---

# 1️⃣ Vertical Data

## What it looks like

Imagine a piece of graph paper. We draw **three vertical stripes** (like three columns). Each stripe is one **class** (e.g. red, green, blue). Points in the left stripe are class 0, in the middle class 1, in the right class 2.

```
     x₁ (horizontal)
     ←———————————————————————————→
  ↑   │  🔴🔴🔴  │  🟢🟢🟢  │  🔵🔵🔵  │
  │   │  🔴🔴🔴  │  🟢🟢🟢  │  🔵🔵🔵  │
x₂   │  🔴🔴🔴  │  🟢🟢🟢  │  🔵🔵🔵  │
  │   │  class 0 │  class 1 │  class 2 │
  ↓   │  (c=0)   │  (c=1)   │  (c=2)   │
     -1.5        0         1.5
```

So: **only the horizontal position (x₁) matters** for the class; the vertical position (x₂) is random noise. A line (or two vertical lines) can separate the three classes → **easy** for the model once it learns the right weights.

---

## The code (simple version)

```python
def vertical_data(samples_per_class=100, classes=3):
    X = np.zeros((samples_per_class * classes, 2))
    y = np.zeros(samples_per_class * classes, dtype=int)
    for c in range(classes):
        ix = range(samples_per_class * c, samples_per_class * (c + 1))
        X[ix, 0] = np.random.randn(samples_per_class) * 0.2 + (c - 1) * 1.5  # vertical bands
        X[ix, 1] = np.random.randn(samples_per_class) * 0.2
        y[ix] = c
    return X, y
```

---

## Line-by-line

### 🎯 “We’ll draw one stripe at a time”

**`for c in range(classes):`**

- `classes=3` → we do 3 stripes: `c = 0`, then `1`, then `2`.
- Each stripe is one “color” (class).

---

### 📍 “Which rows in the big list belong to this stripe?”

**`ix = range(samples_per_class * c, samples_per_class * (c + 1))`**

- Each stripe gets **100 rows** (if `samples_per_class=100`).
- **c=0:** rows 0–99  
- **c=1:** rows 100–199  
- **c=2:** rows 200–299  

So we never mix different stripes in the same slice.

---

### ↔️ “Horizontal position: put this stripe at the right place”

**`X[ix, 0] = np.random.randn(samples_per_class) * 0.2 + (c - 1) * 1.5`**

- **`np.random.randn(samples_per_class)`** → 100 random numbers (mean 0, spread 1).
- **`* 0.2`** → small wiggle so the stripe isn’t a perfect line.
- **`+ (c - 1) * 1.5`** → **where the stripe sits** on the x₁ axis:
  - **c=0:** center at **(0−1)×1.5 = −1.5** (left stripe).
  - **c=1:** center at **(1−1)×1.5 = 0** (middle).
  - **c=2:** center at **(2−1)×1.5 = 1.5** (right).

So we get three vertical bands at x₁ ≈ −1.5, 0, and 1.5.

---

### ↕️ “Vertical position: same random spread for everyone”

**`X[ix, 1] = np.random.randn(samples_per_class) * 0.2`**

- Same small random spread for the **second feature** (x₂).
- No `+ (c - 1) * 1.5` here → all stripes share the same vertical range.
- So **only x₁** tells you which class; x₂ is just noise.

---

### 🏷️ “Remember which stripe each point belongs to”

**`y[ix] = c`**

- For the 100 rows we just filled, we set the label to **c** (0, 1, or 2).
- So **X** = (x₁, x₂) positions, **y** = class index.

---

## Summary diagram (vertical)

```
  x₂
   │     ·  ·  ·     ·  ·  ·     ·  ·  ·
   │   ·  ·  ·  ·   ·  ·  ·  ·   ·  ·  ·  ·
   │  ·  ·  ·  ·  · ·  ·  ·  ·  · ·  ·  ·  ·  ·
   │ —————————————————————————————————————————→ x₁
        -1.5      0       1.5
        class 0  class 1  class 2
```

**In one sentence:** We make three vertical bands (left, middle, right) with a bit of random scatter; the class is decided by horizontal position only, so the problem is easy for the network once it learns the right weights.

---

# 2️⃣ Spiral Data (fallback)

## What it looks like

Imagine drawing **three spiral arms** on paper, like a pinwheel or a simple galaxy. Each arm is one class. We put dots along each arm; the computer’s job is to learn “this point is on the red spiral,” “this one is on the green spiral,” etc.

```
              🟢
         🔵      🔴
       🔵   ·   🔴
      🔵  · · ·  🔴
       🔵  ···  🔴
         🔵 · 🔴
            ·
         (center)
```

The arms **wind around** the center and **overlap** in space (same x,y can belong to different classes depending on which arm they’re on). So you can’t separate them with a single straight line → **hard** for simple or random strategies.

---

## The code (simple version)

```python
def spiral_data_fallback(samples=100, classes=3):
    np.random.seed(42)
    n = samples
    t = np.linspace(0, 4 * np.pi, n)
    X = np.zeros((n * classes, 2))
    y = np.zeros(n * classes, dtype=int)
    for c in range(classes):
        r = np.linspace(0.5, 2, n) + 0.2 * np.random.randn(n)
        X[c * n:(c + 1) * n, 0] = r * np.cos(t + c * 2 * np.pi / classes)
        X[c * n:(c + 1) * n, 1] = r * np.sin(t + c * 2 * np.pi / classes)
        y[c * n:(c + 1) * n] = c
    return X, y
```

---

## Line-by-line

### 🎲 “Same random drawing every time”

**`np.random.seed(42)`**

- When we add small random wiggles, we get the **same** wiggles every run.
- Like fixing “random” in a game so the level is identical each time.

---

### 📐 “How many dots per spiral?”

**`n = samples`**

- Usually 100. So we put **100 dots** on each spiral arm (each class).

---

### 🕐 “Where along the spiral (angle)?”

**`t = np.linspace(0, 4 * np.pi, n)`**

- We pick **100 angles** from 0 to **4π** (about two full turns around the center).
- So we walk “along” the spiral from the center outward in 100 steps.

---

### 📋 “Big list for positions and labels”

**`X = np.zeros((n * classes, 2))`**  
- 300 rows (3 classes × 100 dots). Each row = one point’s **(x, y)**.

**`y = np.zeros(n * classes, dtype=int)`**  
- 300 labels: 0, 1, or 2 (which spiral the point is on).

---

### 🔄 “Draw one spiral at a time”

**`for c in range(classes):`**

- Draw spiral 0, then 1, then 2.
- Each spiral is rotated so the three arms are **120° apart** (like a peace sign or Mercedes logo).

---

### 📏 “How far from the center?”

**`r = np.linspace(0.5, 2, n) + 0.2 * np.random.randn(n)`**

- **Distance from center** for each of the 100 dots:
  - Start near the center (0.5), end farther out (2).
  - **`+ 0.2 * np.random.randn(n)`** → small random wiggle so the spiral isn’t perfectly smooth.

---

### 🕐 “From angle + distance to (x, y)”

**`X[..., 0] = r * np.cos(t + c * 2 * np.pi / classes)`**  
**`X[..., 1] = r * np.sin(t + c * 2 * np.pi / classes)`**

- We use the same **angle steps** `t`, but **rotate** each spiral by **`c * 2π/3`** so the three arms are 120° apart.
- **cos** and **sin** turn “angle + distance” into **x** and **y** (like the hand of a clock: cos → horizontal, sin → vertical).
- So for each class we get 100 points along one spiral arm.

---

### 🏷️ “Which spiral is this?”

**`y[c * n:(c + 1) * n] = c`**

- For the 100 rows we just filled, the label is **c** (0, 1, or 2).

---

### 📤 “Send back points and labels”

**`return X, y`**

- **X:** all (x, y) positions (e.g. 300 points).
- **y:** for each point, 0, 1, or 2.

---

## Summary diagram (spiral)

```
        class 1 (e.g. green)
              ╱
            ╱ ·
          ╱ · ·
        · · ·
      · · ·     class 0 (e.g. red)
    · · ·     ╱
  · · ·     ╱
   · ·     ╱
    ·     · · ·
     ╲   · · ·   class 2 (e.g. blue)
       ╲ · · ·
         ╲ · ·
           ╲
```

**In one sentence:** We draw three spiral arms (100 dots each), spread 120° apart with a little random noise, and return their (x, y) positions and which arm each dot is on, so the computer can learn to tell the spirals apart—and we see that naive “random adjust” optimization struggles here.

---

# 🧩 How they’re used in optimization

1. **Vertical data**  
   - Easy to separate by x₁.  
   - Strategy 1 (fully random weights) still does poorly.  
   - Strategy 2 (random small adjustments, keep improvements) can get good loss and high accuracy.  
   → Shows that “local” random search can work on a simple problem.

2. **Spiral data**  
   - Hard: no single line can separate the classes.  
   - Strategy 2 (same “random adjust”) barely improves loss and accuracy.  
   → Shows we need **gradient-based** updates (direction and step size from the loss), not just random steps.

---

# 📚 Quick reference

| Concept        | Vertical data              | Spiral data                    |
|----------------|----------------------------|--------------------------------|
| **Shape**      | Three vertical bands       | Three spiral arms              |
| **Separation** | By x₁ (horizontal)         | By angle + radius (winding)   |
| **Difficulty** | Easy (linear-ish)          | Hard (non-linear)              |
| **Strategy 1** | Poor                       | —                              |
| **Strategy 2** | Good                       | Poor                           |
| **Need**       | See that something works   | See that we need gradients     |

If you want to try the same ideas with the official NNFS spiral (same idea, possibly different default shape), you can use `nnfs.datasets.spiral_data` when the `nnfs` package is installed; the fallback above works without it and is explained in this document.
