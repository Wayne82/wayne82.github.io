---
layout: post
title:  "Gimbal Lock: Understanding the Problem"
categories: graphics
mathjax: true
comments: true
---

It's been several years since I first encountered the Gimbal Lock problem, and back then, I didn't quite grasp its underlying mechanics. However, after revisiting the issue more recently, I believe I’ve finally cracked the parts that had confused me for so long. In this blog, I want to share my understanding of Gimbal Lock majorly from a mathematical perspective.

## What is Gimbal Lock?
Gimbal Lock is a phenomenon that occurs in systems where rotation is described using three angles, often referred to as Euler angles. These systems are widely used in applications like aviation, robotics, and computer graphics to define orientation in three-dimensional space. Gimbal Lock happens when two of the three rotation axes in such a system become aligned, resulting in the loss of one degree of freedom. This means the system can no longer uniquely represent all possible orientations, leading to ambiguity and unexpected behavior.

The term “Gimbal Lock” can be misleading because none of the individual gimbals are actually restrained. All three gimbals can still rotate freely about their respective axes of suspension. The expression `two of the three rotation axes in such a system become aligned` can also be confusing. At first, it made me question: how can two perpendicular axes become aligned? Moreover, since any rotation in 3D space can be achieved using three Euler angles, how could the lock even occur?

As I sought to answer these questions by exploring various learning materials - including YouTube videos, Wikipedia articles, and even asking ChatGPT — I began to realize there were many key concepts I needed to understand first. These concepts gradually surfaced as I delved deeper, and I will introduce them in the following section. But before that, let me outline two key observations that form the foundation of my understanding:

- **In a motionless state**: When we get a **rotation matrix** by defining a rotation using three Euler angles, if the second angle becomes 90° or -90° and remains fixed, the rotational effect of the first and third axes degrades into behaving like rotations about the same axis. This effect requires some imagination to "visualize" mentally, whether using intrinsic or extrinsic rotations. This is referred to as the `loss of one degree of freedom.` See the next section for the mathematical explainations. 

- **In a motion state**: When animating a transition between an object's orientation A and orientation B by interpolating between keyframes using Euler angles, the `loss of one degree of freedom` caused by Gimbal Lock can lead to unexpected, abrupt "jumps" or rotations. This happens because the **Jacobian matrix** derived from the rotation matrix becomes "singular" (its rank is less than 3). The Jacobian matrix represents how small changes in the Euler angles affect the resulting rotation matrix. When it becomes singular, it indicates that the system's sensitivity to changes in the angles has broken down, leading to unpredictable interpolation paths.

## A Mathematical Perspective on Gimbal Lock
Let's dive deep into the mathematical mechanics behind the phenomenon of `loss of one degree of freedom`.

Suppose we use the **Tait–Bryan** convention for Euler angles, with one possible sequence of rotation axes denoted as x-y-z and corresponding angles (α, β, γ). In the **intrinsic rotation** convention, the final rotation matrix represents the composition of three elemental rotations about the axes $x-y'-z''$, where the $y'$ and $z''$ axes represent the transformed axes after each previous rotation. However, it is not as simple as just multiplying the three elemental rotation matrices in the order the rotations are applied, because the coordinate systems or reference frames in which the elemental rotations are defined are different. Instead, to compute the final rotation matrix, the process can be converted into an **extrinsic rotation** composition. In **extrinsic rotations** convention, all rotations occur about the axes of the same fixed coordinate system, allowing the elemental rotation matrices to be multiplied directly. Additionally, any extrinsic rotation is equivalent to an intrinsic rotation with the same angles but with the order of the elemental rotations reversed (and vice versa). Furthermore, the other conventions discussed in this article apply to **active** rotations of vectors in a **right-hand** coordinate system, performed counterclockwise, and represented by **pre-multiplication** of the rotation matrix.

Then, let's examine the two equivalent rotation matrices, each composed using different rotation conventions.

Intrinsic rotation convention:

$$R = Z''(γ)Y'(β)X(α)$$

Where, X is the elemental rotation around x axis which initially aligns with the fixed coordinate system. $Y'$ and $Z''$ are the elemental rotation around $y'$ and $z''$ axis respectively, but both are expressed in the initial fixed coordinate system.

Extrinsic rotation convention:

$$R = X(α)Y(β)Z(γ)$$

Where, X, Y and Z are all the elemental rotations around the 3 cardinal axes of the initial fixed coordinate system. The proof of the conversion from intrinsic rotation to extrinsic rotation, which results in the inversed order of the multiplication, will be given in the following key concepts section.

Now, we can calculate the rotation matrix given the equation of using the extrinsic rotation convention.

$$
\begin{align}
R & = \begin{bmatrix}
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
\end{bmatrix} \\
& = \begin{bmatrix}
\cos \beta \cos \gamma & -\cos \beta \sin \gamma & \sin \beta \\
\cos \alpha \sin \gamma + \cos \gamma \sin \alpha \sin \beta & \cos \alpha \cos \gamma - \sin \alpha \sin \beta \sin \gamma & -\cos \beta \sin \alpha \\
\sin \alpha \sin \gamma - \cos \alpha \cos \gamma \sin \beta & \sin \alpha \cos \gamma + \cos \alpha \sin \beta \sin \gamma & \cos \alpha \cos \beta
\end{bmatrix}
\end{align}
$$

When Gimbal Lock happens, the Eular Angle $\beta=\pm\frac{\pi}{2}$. Without loss of generality, let's assume $\beta=+\frac{\pi}{2}$. Knowing that $\cos \frac{\pi}{2}=0$ and $\sin \frac{\pi}{2}=1$, the above expression becomes

$$
R = \begin{bmatrix}
0 & 0 & 1 \\
\sin(\alpha + \gamma) & \cos(\alpha + \gamma) & 0 \\
-\cos(\alpha + \gamma) & \sin(\alpha + \gamma) & 0
\end{bmatrix}
$$

The key obervations from this result are that the angles $\alpha$, and $\gamma$ now appear only in the combination $\alpha + \gamma$, and the third column is fixed $\begin{bmatrix} 1 & 0 & 0 \end{bmatrix}^T$. This means:
- The rotations about the z axis ($\gamma$) and x axis ($\alpha$) are no longer independent. They effectively, as shown in the calculated rotation matrix, collapse into one single axis $\begin{bmatrix} 1 & 0 & 0 \end{bmatrix}^T$.
- Instead of having 3 independent paramters $(\alpha, \beta, \gamma)$ representing the full freedom, the system now depends on only 2 parameters: $\beta = 90^\circ$ and $\alpha + \gamma$, which concludes exactly `loss of one degree of freedom`.

The Jacobian matrix $J$ is a $9 \times 3$ matrix, where each element represents the partial derivative of the corresponding element of the rotation matrix $R$ in a row-major format with respect to the Euler angles $\alpha$, $\beta$, and $\gamma$. I won't delve too deeply into the mathematical details of the Jacobian matrix, as I am not entirely confident I fully understand it myself right now. However, what we can infer is that the "singularity" of the Jacobian matrix when Gimbal Lock occurs leads to a degenerative response to small changes in the Euler angles. This could result in sudden, unpredictable behavior, such as discontinuities in interpolation when animating rotations.

## The Key Concepts
Without expanding into too much details, I want to touch on the key concepts I've learnt so as to reason about the Gimbal Lock problem.

### Euler Angles
Main resource: Wikipedia - [Euler Angles](https://en.wikipedia.org/wiki/Euler_angles)

A set of three angles introduced by Euler that describe the orientation of a rigid body in 3D space by sequentially rotating it about different axes.
- Euler angle defines 3 elemental rotations (rotations about the axes of a coordinate system), which composing together can represent any rotation in 3D space.
- The three elemental rotations may be extrinsic (rotations about the axes xyz of the original coordinate system, which is assumed to remain motionless), or intrinsic (rotations about the axes of the rotating coordinate system XYZ, solidary with the moving body, which changes its orientation with respect to the extrinsic frame after each elemental rotation)
- The order of the rotation matters, thus there exist twelve possible sequences of rotation axes, divided in two groups: **Proper Euler angles** and **Tait–Bryan angles**
- Different authors may use different sets of rotation axes to define Euler angles, or different names for the same angles. So, any discussion employing Euler angles should always be preceded by their definition.

### Reference Frame
A coordinate system used to describe the position and orientation of objects. Transformations and rotations can be interpreted differently depending on whether they are applied relative to a moving (body) frame or a fixed frame.
- The coordinate system is commonly defined using right hand rule.
- Transformation or the rotation matrix defined in different reference frame can't be multiplied directly.

### Active and Passive Transformation
Main resource: Wikipedia - [Active and passive transformation](https://en.wikipedia.org/wiki/Active_and_passive_transformation)

An active transformation rotates or moves an object while keeping the coordinate system fixed, whereas a passive transformation changes the coordinate system while keeping the object unchanged. 
- These are mathematically equivalent but conceptually different interpretations of transformations.
- The actual rotation matrix applied to an object for a passive transformation is the inverse (or transpose) of the rotation matrix applied in an active transformation.

### Rotation Matrix
Main resource: Wikipedia - [Rotation Matrix](https://en.wikipedia.org/wiki/Rotation_matrix)

Rotation matrix is a transformation matrix that is used to perform a rotation in 3D Euclidean space.
- A rotation matrix can be constructed using Euler angles, axis-angle representation, or quaternions. 
- Rotation matrix multiplication is not commutative. Given a column vector to rotate, the order of rotation operations is from right to left.
- It satisfies the orthogonality condition: $R^T R = I$, meaning its inverse is its transpose. And its determinant is 1.
- A rotation preserves both distances between points and handedness.

### Intrinsic and Extrinsic Rotations
Main resource: Wikipedia - [Davenport chained rotations](https://en.wikipedia.org/wiki/Davenport_chained_rotations)

Intrinsic rotation means the object rotates relative to its own moving coordinate frame, where the axes change after each rotation. Extrinsic rotation means the object rotates about fixed coordinate axes, maintaining the same reference frame throughout. Any intrinsic rotation sequence has an equivalent extrinsic rotation sequence with reversed order.

The proof of conversion from intrinsic rotation to extrinsic rotation can be found in [this wikipedia article](https://en.wikipedia.org/wiki/Davenport_chained_rotations#The_proof_of_the_conversion_in_the_pre-multiply_case). However, the frame notation used in the proof is unfamiliar to me and I do not fully get its usage in the proof. Therefore, I developed my own interpretation of the proof, which I outline below.

We aim to prove that

$$
R = Z'' Y'X = XYZ
$$

where R is the intrinsic rotation sequence expressed in the fixed frame.

First, computing $Y'$,

$$
Y' = XYX^{-1}
$$

$Y'$ is the intrinsic elemental rotation matrix around $y'$ axis which is rotated after the first rotation around $x$ axis, and represented in the fixed frame. The reasoning behind this equation is as follows,
- First, apply $X^{-1}$ to reverse the first elemental rotation $X$, aligning $y'$ back with the original $y$ axis in the fixed frame.
- Then, apply the elemental rotation matrix $Y$.
- Finally, re-apply $X$, now with the elemental rotation $Y$ being incorporated into the transformation sequence as well.

Then, computing $Z''$, following the same reasoning,

$$
\begin{align}
Z'' & = Y'XZX^{-1}Y'^{-1} \\
& = (XYX^{-1})XZX^{-1}(XYX^{-1})^{-1} \\
& = XYX^{-1}XZX^{-1}XY^{-1}X^{-1} \\
& = XYZY^{-1}X^{-1}
\end{align}
$$

$Z''$ is the intrinsic elemental rotation matrix around $z''$ axis which is rotated after both the first and second rotation around $x$ and $y'$ axis respectively, and represented in the fixed frame. The inference process is similar as computing $Y'$.

Finally, substituting the expression for $Y'$ and $Z''$, we can get R,

$$
\begin{align}
R & = Z''Y'X \\
& = (XYZY^{-1}X^{-1})(XYX^{-1})X \\
& = XYZY^{-1}(X^{-1}X)Y(X^{-1}X) \\
& = XYZ(Y^{-1}Y) \\
& = XYZ
\end{align}
$$

Thus, we have shown that an intrinsic rotation matrix composition $Z''Y'X$ is mathematically equivalent to an extrinsic rotation matrix composition $XYZ$, which is in reverse order in terms of elemental rotation matrix.

### Jacobian Matrix
Main resource: Wikipedia - [Jacobian Matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)

A matrix of partial derivatives that describes how small changes in input variables (such as Euler angles) affect output variables (such as rotation matrix elements or object orientation). In the context of 3D rotation, the Jacobian helps analyze how singularities, like Gimbal Lock, impact motion smoothness.

## My Learning Process
As I already mentioned at the beginning of the blog, the main resources I used to learn this subject are Wikipedia, Youtube videos, and some assistance from ChatGPT. Many key concepts gradually emerged, helping to shape my understanding. While I believe further reading of formal textbooks in the related fields would deepen my knowledge, the materials I have explored so far have been sufficient to give me a basic grasp of the problem’s main structure. And in this section, I’d like to share two pivotal Aha moments that significantly advanced my understanding of Gimbal Lock during this learing process:

1. **Understanding Intrinsic vs. Extrinsic Rotation Conventions**: Grasping the difference between intrinsic and extrinsic rotation conventions for composing a rotation matrix from three Euler angles—and how to convert between the two—was a crucial insight in understanding the "loss of one degree of freedom" caused by Gimbal Lock. This realization helped clarify why certain rotations become degenerate as the system approaches Gimbal Lock. Specifically, it explained how two axes of rotation can become aligned in terms of their rotational effects (not physically, but in the sense of their combined impact on the object’s orientation), leading to the loss of one degree of freedom.

2. **The Role of the Jacobian Matrix**: Gaining a general understanding of the Jacobian matrix derived from the rotation matrix was another breakthrough. This matrix is key to understanding why Gimbal Lock can cause unpredictable behavior when animating a transition between two orientations of the same object. Specifically, the Jacobian’s "singularity" can lead to discontinuities or abrupt jumps in the animation, as small changes in the Euler angles no longer produce smooth, continuous rotations.

## Final Words
I find it deeply satisfying to grasp this problem from a mathematical perspective, even though the journey often leads to many "rabbit holes" when diving deeper. For example, exploring alternative representations of 3D rotations to avoid Gimbal Lock, such as quaternions, understanding how to extract Euler angles from a rotation matrix, or studying 3D rotations as a mathematical group — specifically, the special orthogonal group $SO(3)$, which is defined as $SO(3) = \{R \in \mathbb{R}^{3 \times 3} \mid R^TR = I, \det(R) = 1\}$, representing proper rotations in three-dimensional Euclidean space, etc. For now, I’ll set these topics aside for future exploration, until another interesting problem sparks a deeper dive. With that, I’ll conclude this blog here.