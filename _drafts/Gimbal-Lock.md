---
layout: post
title:  "Gimbal Lock: Understanding the Problem"
categories: graphics
---

It's been several years since I first encountered the Gimbal Lock problem, and back then, I didn't quite grasp its underlying mechanics. However, after revisiting the issue more recently, I believe I’ve finally cracked the parts that had confused me for so long. In this blog, I want to share my understanding of Gimbal Lock and how it affects systems like computer graphics.

## What is Gimbal Lock?
Gimbal Lock is a phenomenon that occurs in systems where rotation is described using three angles, often referred to as Euler angles. These systems are widely used in applications like aviation, robotics, and computer graphics to define orientation in three-dimensional space. Gimbal Lock happens when two of the three rotation axes in such a system become aligned, resulting in the loss of one degree of freedom. This means the system can no longer uniquely represent all possible orientations, leading to ambiguity and unexpected behavior.

The term “Gimbal Lock” can be misleading because none of the individual gimbals are actually restrained. All three gimbals can still rotate freely about their respective axes of suspension. The expression `two of the three rotation axes in such a system become aligned` can also be confusing. At first, it made me question: how can two perpendicular axes become aligned? Moreover, since any rotation in 3D space can be achieved using three Euler angles, how could the lock even occur?

As I sought to answer these questions by exploring various learning materials - including YouTube videos, Wikipedia articles, and even asking ChatGPT — I began to realize there were many key concepts I needed to understand first. These concepts gradually surfaced as I delved deeper, and I will introduce them in the following section. But before that, let me outline two key observations that form the foundation of my understanding:

- **In a motionless state**: When we define a rotation using three Euler angles, if the **second** angle becomes 90° or -90° and remains fixed, the rotational effect of the first and third axes degrades into behaving like rotations about the same axis. It needs a bit of imagination to "visualize" the effect in head with either intrisic or extrinsic rotation. This is referred to as the `loss of one degree of freedom.`

- **In a motion state**: When animating a transition between an object's orientation A and orientation B by interpolating between keyframes using Euler angles, the `loss of one degree of freedom` caused by Gimbal Lock can lead to unexpected, abrupt "jumps" or rotations. This happens because the Jacobian matrix derived from Euler angles becomes singular. The Jacobian matrix represents how small changes in the Euler angles affect the resulting rotation matrix. When it becomes singular, it indicates that the system's sensitivity to changes in the angles has broken down, leading to unpredictable interpolation paths.

## A Mathematical Perspective on Gimbal Lock

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

