---
layout: post
title:  "Learn Linear Algebra a Different Way"
categories: mathematics
mathjax: true
comments: true
---

It's been a while since my last post about neural networks, which covers certain mathematical details, to my own interest. As originally planned, my next learning topic is attention mechanism and transformers. However, the book of "Linear Algebra Done Right" by Sheldon Axler lying on the table for quite a while caught my attention eventually. And one day, I can't help picking it up and start reading. Until now, I have read the first 7 chapters, and find it is quite a fresh way to learn linear algebra for someone like me who only learned it in college as a general course 2 decades ago. There are 2 more chapters to go, but I think it is good for me to stop at this point and write down some notes about what I have learned so far.

## The Essence of Linear Algebra
Linear algebra is the study of linear structures, which covers:
* The linear space (vector space), which is a set **V** along with an addition on **V** and a scalar multiplication on **V**, satisfying certain axioms: commutativity, associativity, additive identity, additive inverse, multiplicative identity, and distributivity.
* The linear transformation (linear map), which is defined as a function T: **V** → **W** with the property of additivity and homogeneity.
* The linear representation (matrix), which is a way to represent a linear transformation with respect to the bases of the vector spaces involved.
