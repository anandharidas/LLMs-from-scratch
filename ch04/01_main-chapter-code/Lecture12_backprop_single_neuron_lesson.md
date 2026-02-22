# 📘 Lecture 12: Backpropagation from Scratch on a Single Neuron

**Source:** [Lecture 12 - Backpropagation from scratch on a single neuron](https://www.youtube.com/watch?v=iE1lccrHfok) (Vizuara, ~34 min)

This note is the full lesson: one neuron, ReLU, squared loss, and gradient descent via the **chain rule**—no matrices, just scalars and derivatives.

---

## 🎯 What We Want

- **Problem:** Adjust the **weights** and **bias** of a single neuron so its **output gets close to 0** (our target).
- **Tool:** **Gradient descent** — update each parameter by moving a small step in the **negative gradient** direction (steepest descent).
- **How we get gradients:** **Backpropagation** = apply the **chain rule** from the loss back to each parameter.

---

## 1️⃣ The Single-Neuron Setup

### Schematic (forward direction)

One neuron: **3 inputs**, **3 weights**, **1 bias** → **ReLU** → **output Z** → **squared loss** (target = 0).

```
  x₀ ──●── w₀ ──┐
                │
  x₁ ──●── w₁ ──┼──► Σ ──► ReLU ──► Z ──► L = (Z − 0)² = Z²
                │     (sum)  (max(·,0))
  x₂ ──●── w₂ ──┘
                │
  bias b ───────┘
```

### Mathematical form

| Step | Formula |
|------|--------|
| Weighted sum (pre-activation) | \( a = x_0 w_0 + x_1 w_1 + x_2 w_2 + b \) |
| ReLU | \( Z = \text{ReLU}(a) = \max(a,\, 0) \) |
| Loss (target = 0) | \( L = (Z - 0)^2 = Z^2 \) |

**Goal:** Minimize \(L\) by updating \(w_0, w_1, w_2, b\).

---

## 2️⃣ Why Gradient Descent?

- **Bad way:** Try thousands of random \((w_0, w_1, w_2, b)\), keep the one with lowest loss. Inefficient.
- **Good way:** At the current point on the “loss surface,” move in the **direction of steepest decrease** = **negative gradient**.

```
     L (loss)
      │
      │     *
      │    ╱ ╲
      │   ╱   ╲
      │  ╱  •  ╲     ← current (w,b)
      │ ╱   ↓   ╲    ← move opposite to gradient
      │╱    ↓    ╲
      └──────────────→ parameters
              minimum
```

Update rule (learning rate \(\eta\)):

\[
w_0^{\text{new}} = w_0 - \eta \cdot \frac{\partial L}{\partial w_0},\quad
w_1^{\text{new}} = w_1 - \eta \cdot \frac{\partial L}{\partial w_1},\quad
\ldots,\quad
b^{\text{new}} = b - \eta \cdot \frac{\partial L}{\partial b}.
\]

So we need the **four partial derivatives**: \(\frac{\partial L}{\partial w_0},\ \frac{\partial L}{\partial w_1},\ \frac{\partial L}{\partial w_2},\ \frac{\partial L}{\partial b}\). Backpropagation is how we compute them using the **chain rule**.

---

## 3️⃣ Breaking the Forward Pass into Steps (for the chain rule)

Think of the forward pass as a **pipeline** of small steps. Each step has a simple derivative; the chain rule multiplies them.

```
  [x₀·w₀]   [x₁·w₁]   [x₂·w₂]   [b]
      ╲        │        │       │
       ╲       │        │       │
        ╲      ▼        ▼       ▼
         ╲   ┌─────────────────────┐
          ╲  │   SUM = x₀w₀+...+b   │
           ╲ └──────────┬──────────┘
            ╲           ▼
             ╲    ┌─────────┐
              ╲   │  ReLU   │
               ╲  └────┬────┘
                ╲     ▼
                 ╲   Z
                  ╲  │
                   ╲ ▼
                    L = Z²
```

- **Loss** \(L\) depends on **Z**.
- **Z** depends on **ReLU** of the **sum**.
- **Sum** depends on **x₀w₀**, **x₁w₁**, **x₂w₂**, and **b**.
- So \(L\) depends on \(w_0\) only **through** ReLU → sum → (x₀w₀). Same idea for \(w_1, w_2, b\). That’s why we use the **chain rule**.

---

## 4️⃣ Chain Rule for \(\frac{\partial L}{\partial w_0}\)

We want \(\frac{\partial L}{\partial w_0}\). \(L\) depends on \(w_0\) only through this path:

\[
w_0 \;\rightarrow\; (x_0 w_0) \;\rightarrow\; \text{sum} \;\rightarrow\; \text{ReLU} \;\rightarrow\; L
\]

So:

\[
\boxed{
\frac{\partial L}{\partial w_0}
\;=\;
\frac{\partial L}{\partial Z}
\;\cdot\;
\frac{\partial Z}{\partial (\text{sum})}
\;\cdot\;
\frac{\partial (\text{sum})}{\partial (x_0 w_0)}
\;\cdot\;
\frac{\partial (x_0 w_0)}{\partial w_0}
}
\]

**Same pattern for \(w_1\) and \(w_2\):** only the **last** factor changes (it becomes \(x_1\) or \(x_2\)). For **b**, the path is shorter: no “multiplication by input,” so we get three factors (loss → ReLU → sum → b).

---

## 5️⃣ Computing Each Piece (with numbers)

Use the video’s example:

- **Inputs:** \(x_0 = 1,\ x_1 = -2,\ x_2 = 3\)
- **Weights & bias:** \(w_0 = -3,\ w_1 = -1,\ w_2 = 2,\ b = 1\)

### Step A: Forward pass (to get numbers we need)

| Step | Formula | Value |
|------|--------|--------|
| Sum | \(a = x_0 w_0 + x_1 w_1 + x_2 w_2 + b\) | \((-3)+2+6+1 = 6\) |
| ReLU | \(Z = \max(6, 0)\) | \(6\) |
| Loss | \(L = Z^2\) | \(36\) |

So we’ll use **sum = 6**, **Z = 6**, **L = 36** in the derivatives below.

---

### Step B: The four factors for \(\frac{\partial L}{\partial w_0}\)

**1. \(\frac{\partial L}{\partial Z}\)**

\(L = Z^2\) ⇒ \(\frac{\partial L}{\partial Z} = 2Z = 2 \times 6 = 12\).

**2. \(\frac{\partial Z}{\partial (\text{sum})}\)** (ReLU derivative)

ReLU is \(\max(\text{sum}, 0)\). So:
- If sum > 0: \(Z = \text{sum}\) ⇒ derivative = **1**.
- If sum ≤ 0: \(Z = 0\) (constant) ⇒ derivative = **0**.

Here sum = 6 > 0 ⇒ **1**.

**3. \(\frac{\partial (\text{sum})}{\partial (x_0 w_0)}\)**

Sum = \((x_0 w_0) + (x_1 w_1) + (x_2 w_2) + b\). Derivative of sum w.r.t. one of its terms is **1**. So **1**.

**4. \(\frac{\partial (x_0 w_0)}{\partial w_0}\)**

Treat \(x_0\) as constant ⇒ \(\frac{\partial}{\partial w_0}(x_0 w_0) = x_0 = 1\). So **1**.

---

### Step C: Put them together

\[
\frac{\partial L}{\partial w_0}
\;=\;
12 \times 1 \times 1 \times 1 \;=\; 12
\]

Diagram of the “backward” chain (gradients flow right to left):

```
  dL/dw₀  ←──  (×1)  ←──  (×1)  ←──  (×1)  ←──  12
   = 12         x₀         ∂sum/∂(x₀w₀)     dL/dZ
```

---

## 6️⃣ Gradients for \(w_1\), \(w_2\), and \(b\)

- **First three factors** are the same for every weight: \(12,\ 1,\ 1\).
- **Fourth factor** for \(w_i\) is the **input** for that weight: \(\frac{\partial (x_i w_i)}{\partial w_i} = x_i\).

So:

| Parameter | Fourth factor | Gradient |
|-----------|----------------|----------|
| \(w_0\) | \(x_0 = 1\) | \(12 \times 1 \times 1 \times 1 = 12\) |
| \(w_1\) | \(x_1 = -2\) | \(12 \times 1 \times 1 \times (-2) = -24\) |
| \(w_2\) | \(x_2 = 3\) | \(12 \times 1 \times 1 \times 3 = 36\) |

For **bias** \(b\): path is Loss → ReLU → sum → b. So only three factors: \(12 \times 1 \times 1 = 12\) (derivative of sum w.r.t. \(b\) is 1).

\[
\frac{\partial L}{\partial b} = 12
\]

---

## 7️⃣ One Gradient Descent Step (with \(\eta = 0.01\))

\[
\begin{aligned}
w_0^{\text{new}} &= w_0 - \eta \cdot \frac{\partial L}{\partial w_0} = -3 - 0.01 \times 12 = -3.12 \\
w_1^{\text{new}} &= -1 - 0.01 \times (-24) = -0.76 \\
w_2^{\text{new}} &= 2 - 0.01 \times 36 = 1.64 \\
b^{\text{new}} &= 1 - 0.01 \times 12 = 0.88
\end{aligned}
\]

New sum ≈ \(5.82\), new Z ≈ \(5.82\), new loss ≈ **33.88** (down from 36). Repeating many steps (e.g. 200) with the same rule drives the loss toward 0.

---

## 8️⃣ Why “Back” Propagation?

We **start from the loss** (right) and **go back** toward the parameters (left), multiplying derivatives at each step:

```
  L ←── Z ←── sum ←── (x₀w₀, x₁w₁, x₂w₂, b)
  │      │      │              │
  dL/dZ  dZ/d(sum)  d(sum)/d(·)  d(xᵢwᵢ)/dwᵢ
```

So we **propagate** the gradient **backward** through the graph. That’s backpropagation.

---

## 9️⃣ Summary Diagram (one neuron)

```
  FORWARD (left → right):
  x₀,w₀,x₁,w₁,x₂,w₂,b → sum → ReLU → Z → L = Z²

  BACKWARD (right → left) for ∂L/∂w₀:
  ∂L/∂w₀ = (∂L/∂Z) · (∂Z/∂sum) · (∂sum/∂(x₀w₀)) · (∂(x₀w₀)/∂w₀)
           =  2Z    ·    1      ·        1        ·      x₀

  UPDATE:
  w₀ := w₀ − η · ∂L/∂w₀   (and similarly w₁, w₂, b)
```

---

## 🔟 Key Takeaways

1. **Single neuron:** inputs × weights + bias → ReLU → Z; loss = Z² (target 0).
2. **Gradient descent:** update each parameter by subtracting \(\eta \times\) (partial derivative of \(L\)).
3. **Backprop = chain rule:** break the forward pass into steps; multiply derivatives along the path from \(L\) to each parameter.
4. **ReLU derivative:** 1 if input > 0, 0 otherwise.
5. **Last factor for weight \(w_i\):** the corresponding input \(x_i\).
6. Doing this for **one neuron** is the core of backprop; layers of many neurons just repeat and combine this idea.

The Jupyter notebook **Lecture12_backprop_single_neuron.ipynb** implements the same forward pass, backward pass, and updates with clear step-by-step iterations and printed values.
