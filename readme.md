# Understanding the Z-Level Image Variations in Synthetic Holography

This document explains, in mathematical and signal-processing terms, what happens in the three approaches (A, B, and C) used to make holographic images vary visibly with propagation distance **z**.

---

## 🟦 (A) Zoom Scaling — Synthetic Spatial Rescaling

### 🔹 What We Did

We applied a **zoom (spatial scaling)** directly to the *intensity image* after propagation:

$$
I_z(x, y) = |U_z(x, y)|
$$

$$
I'_z(x, y) = I_z\left(\frac{x}{s_z}, \frac{y}{s_z}\right)
$$

with a scale factor

$$
s_z = 1 + \frac{z}{z_{\text{max}}} \cdot 0.5
$$

At the farthest distance (\( z = 100\,\mu m \)), the image is scaled by 1.5×.

---

### 🔹 Signal-Processing Interpretation

This is a **spatial-domain resampling** operation.

Scaling in space contracts the frequency spectrum:

$$
\mathcal{F}\{I'(x, y)\}(f_x, f_y) = \frac{1}{s_z^2}\, \mathcal{F}\{I(x, y)\}\left(\frac{f_x}{s_z}, \frac{f_y}{s_z}\right)
$$

This is a **magnification transform**, not a physical propagation.

---

### 🔹 Physical Interpretation

Real diffraction introduces slight magnification, but it’s usually small.  
Here, we **exaggerate** that effect to make Z-levels visibly distinct.

✅ *Perceptually valid*  
⚠️ *Not physically exact*

---

## 🟩 (B) Phase Defocus — Quadratic Phase Modulation

### 🔹 What We Did

We modulated the complex amplitude with a **quadratic phase** before propagation:

$$
U_z(x, y) = A(x, y) \cdot \exp\left(i\, \pi \frac{x^2 + y^2}{\lambda z}\right)
$$

and then propagated using the angular-spectrum operator.

---

### 🔹 Signal-Processing Interpretation

The quadratic phase multiplies the field by a **parabolic phase ramp**,  
which in Fourier optics acts as a **convolution with a defocus kernel**:

$$
H_{\text{defocus}}(f_x, f_y) = \exp\left(-i\, \pi\, \lambda z\, (f_x^2 + f_y^2)\right)
$$

This is the standard **Fresnel propagation kernel**.

---

### 🔹 Physical Interpretation

This simulates **optical defocus** — as if changing the lens focus of the imaging system.

✅ *Physically meaningful*  
Produces realistic defocus blur and focus variation with depth.

---

## 🟨 (C) Combined Zoom + Defocus — Magnification + Wavefront Curvature

### 🔹 What We Did

We first applied the quadratic phase (defocus), **then** zoomed the resulting intensity image:

$$
I'_z(x, y) = \left| \left[ A(x, y)\, e^{\, i \pi (x^2 + y^2)/(\lambda z)} \otimes h_z(x, y) \right] \right|_{\text{scaled}}
$$

where \( \otimes \) denotes convolution and \( h_z \) is the propagation kernel.

---

### 🔹 Signal-Processing Interpretation

This combines two linear operations:

1. **Multiplication by a chirp** → phase curvature (defocus)  
2. **Resampling** → geometric scaling (zoom)

Each operation is linear and approximately energy-preserving.

---

### 🔹 Physical Interpretation

In real optics, propagation over distance naturally causes:

- Phase curvature (defocus)
- Slight magnification (geometric spreading)

The angular-spectrum model already includes both slightly;  
we amplify them for visual clarity.

⚙️ *Semi-physical, perceptually faithful*

---

## ✅ Summary Table

| Version | Mathematical Operation | Signal-Processing Meaning | Physical Validity | Visual Effect |
|----------|------------------------|----------------------------|-------------------|----------------|
| **A – Zoom scaling** | \( I'(x,y) = I(x/s, y/s) \) | Resampling / interpolation | ✖️ Synthetic | Magnification / zoom |
| **B – Phase defocus** | \( U'(x,y) = A(x,y)e^{i\pi(x^2+y^2)/(\lambda z)} \) | Quadratic phase modulation = optical defocus | ✔️ Physical | Progressive blur / focus shift |
| **C – Combined** | Defocus + Resampling | Chirp multiplication + spatial scaling | ⚙️ Semi-physical | Zoom + blur (depth realism) |

---

## 🧠 Summary

- **A** adds visible depth by geometric scaling — good for visual dataset separation.  
- **B** simulates real optical focus shift — physically grounded.  
- **C** combines both — most realistic visually and perceptually.

If your goal is **training data for machine learning**, use **B** or **C**.  
If your goal is **visual differentiation** for humans, **A** or **C** will work best.
