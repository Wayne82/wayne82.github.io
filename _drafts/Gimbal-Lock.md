---
layout: post
title:  "Gimbal Lock: Understanding the Problem"
categories: graphics
mathjax: true
---

It's been several years since I first encountered the Gimbal Lock problem, and back then, I didn't quite grasp its underlying mechanics. However, after revisiting the issue more recently, I believe I’ve finally cracked the parts that had confused me for so long. In this blog, I want to share my understanding of Gimbal Lock and how it affects systems like computer graphics.

## What is Gimbal Lock?
Gimbal Lock is a phenomenon that occurs in systems where rotation is described using three angles, often referred to as Euler angles. These systems are widely used in applications like aviation, robotics, and computer graphics to define orientation in three-dimensional space. Gimbal Lock happens when two of the three rotation axes in such a system become aligned, resulting in the loss of one degree of freedom. This means the system can no longer uniquely represent all possible orientations, leading to ambiguity and unexpected behavior.

The term “Gimbal Lock” can be misleading because none of the individual gimbals are actually restrained. All three gimbals can still rotate freely about their respective axes of suspension. The expression `two of the three rotation axes in such a system become aligned` can also be confusing. At first, it made me question: how can two perpendicular axes become aligned? Moreover, since any rotation in 3D space can be achieved using three Euler angles, how could the lock even occur?

As I sought to answer these questions by exploring various learning materials - including YouTube videos, Wikipedia articles, and even asking ChatGPT — I began to realize there were many key concepts I needed to understand first. These concepts gradually surfaced as I delved deeper, and I will introduce them in the following section. But before that, let me outline two key observations that form the foundation of my understanding:

- **In a motionless state**: When we define a rotation using three Euler angles, if the **second** angle becomes 90° or -90° and remains fixed, the rotational effect of the first and third axes degrades into behaving like rotations about the same axis. This effect requires some imagination to "visualize" mentally, whether using intrinsic or extrinsic rotations. This is referred to as the `loss of one degree of freedom.`

- **In a motion state**: When animating a transition between an object's orientation A and orientation B by interpolating between keyframes using Euler angles, the `loss of one degree of freedom` caused by Gimbal Lock can lead to unexpected, abrupt "jumps" or rotations. This happens because the Jacobian matrix derived from Euler angles becomes singular. The Jacobian matrix represents how small changes in the Euler angles affect the resulting rotation matrix. When it becomes singular, it indicates that the system's sensitivity to changes in the angles has broken down, leading to unpredictable interpolation paths.

## A Mathematical Perspective on Gimbal Lock
Let's dive deep into the mathematical mechanics behind the phenomenon of `loss of one degree of freedom`.

Suppose we use the **Tait–Bryan** convention for Euler angles, with one possible sequence of rotation axes denoted as x-y-z and corresponding angles (α, β, γ). In the **intrinsic rotation** convention, the final rotation matrix represents the composition of three elemental rotations about the axes $x-y'-z''$, where the $y'$ and $z''$ axes represent the transformed axes after each previous rotation. However, it is not as simple as just multiplying the three elemental rotation matrices in the order the rotations are applied, because the coordinate systems or reference frames in which the elemental rotations are defined are different. Instead, to compute the final rotation matrix, the process can be converted into an **extrinsic rotation** composition. In **extrinsic rotations** convention, all rotations occur about the axes of the same fixed coordinate system, allowing the elemental rotation matrices to be multiplied directly. Additionally, any extrinsic rotation is equivalent to an intrinsic rotation with the same angles but with the order of the elemental rotations reversed (and vice versa). Furthermore, the other conventions discussed in this article apply to **active** rotations of vectors in a **right-handed** coordinate system, performed counterclockwise, and represented by **pre-multiplication** of the rotation matrix.

Then, let's examine the two equivalent rotation matrices, each composed using different rotation conventions.

Intrinsic rotation convention:

$$R = Z''(γ)Y'(β)X(α)$$

Where, X is the elemental rotation around x axis which initially aligns with the fixed coordinate system. $Y'$ and $Z''$ are the elemental rotation around $y'$ and $z''$ axis respectively, but both are expressed in the initial fixed coordinate system.

Extrinsic rotation convention:

$$R = X(α)Y(β)Z(γ)$$

Where, X, Y and Z are all the elemental rotations around the 3 principal axes of the initial fixed coordinate system. The proof of the conversion from intrinsic rotation to extrinsic rotation, which results in the inversed order of the multiplication, will be given in the following key concepts section.

Now, we can calculate the rotation matrix given the equation of using the extrinsic rotation convention.

$$
R = \begin{bmatrix}
1 & 0 & 0 \\
0 & \cos \alpha & -\sin \alpha \\
0 & \sin \alpha & \cos \alpha
\end{bmatrix}
\cdot
\begin{bmatrix}
\cos \beta & 0 & \sin \beta \\
0 & 1 & 0 \\
-\sin \beta & 0 & \cos \beta
\end{bmatrix}
\cdot
\begin{bmatrix}
\cos \gamma & -\sin \gamma & 0 \\
\sin \gamma & \cos \gamma & 0 \\
0 & 0 & 1
\end{bmatrix}
= \begin{bmatrix}
\cos \beta \cos \gamma & -\cos \beta \sin \gamma & \sin \beta \\
\cos \alpha \sin \gamma + \cos \gamma \sin \alpha \sin \beta & \cos \alpha \cos \gamma - \sin \alpha \sin \beta \sin \gamma & -\cos \beta \sin \alpha \\
\sin \alpha \sin \gamma - \cos \alpha \cos \gamma \sin \beta & \sin \alpha \cos \gamma + \cos \alpha \sin \beta \sin \gamma & \cos \alpha \cos \beta
\end{bmatrix}
$$

When Gimbal Lock happens, the Eular Angle $\beta=\pm\frac{\pi}{2}$. Without loss of generality, let's assume $\beta=+\frac{\pi}{2}$. Knowing that $\cos \frac{\pi}{2}=0$ and $\sin \frac{\pi}{2}=1$, the above expression becomes

$$
R = \begin{bmatrix}
0 & 0 & 1 \\
\sin(\alpha + \gamma) & \cos(\alpha + \gamma) & 0 \\
-\cos(\alpha + \gamma) & \sin(\alpha + \gamma) & 0
\end{bmatrix}
$$

## Real World Implications of Gimbal Lock
- **Computer Graphics**
- **Others**

## The Key Concepts

- **Euler angles** (geometrical definition, proper and Tait-Bryan angles)
- **Frame of reference** (handedness)
- **Active and passive** transformation
- **Rotation Matrix** 

  apply to active rotations of vectors counterclockwise in a right-handed coordinate system (y counterclockwise from x) by pre-multiplication (R on the left) of the column vectors.

- **Intrinsic and Extrinsic Rotations**
- **Jacobian Matrix**

## My Learning Process

## Final Words

