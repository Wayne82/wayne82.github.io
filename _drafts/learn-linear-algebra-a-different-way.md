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
* **The linear space (vector space)**, which is a set **V** along with an addition on **V** and a scalar multiplication on **V**, satisfying certain axioms: commutativity, associativity, additive identity, additive inverse, multiplicative identity, and distributivity.
  * Examples: **R**^n, polynomials, etc.
* **The linear transformation (linear map)**, which is defined as a function T: **V** → **W** with the property of additivity and homogeneity.
  * $$T(u + v) = T(u) + T(v)$$ and $$T(cv) = cT(v)$$ for all vectors u, v in **V** and all scalars c.
  * Examples: differentiation, integration, etc.
* **The linear representation (matrix)**, which is a way to represent a linear transformation with respect to the bases of the vector spaces involved.
  * Linear maps can be represented by matrices once you choose a basis for the domain and codomain vector spaces.
  * Changing the basis gives different representations, but the underlying transformation is the same!
  * Much of linear algebra is about finding the "best" basis to understand a map (diagonalization, orthogonal bases).
  * Examples: rotation matrix, projection matrix, etc.
* **The structure and invariants**, which are the properties of linear spaces and linear transformations that remain unchanged under certain operations, such as dimension, rank, nullity, eigenvalues, and eigenvectors.

It is less about solving linear equations in the traditional sense, but more about understanding the geometry and structure of linear systems, how they transform and what invariants they preserve.

## The Key Definitions
Here I will simply record the key definitions I have learned so far, as a quick reference for myself.
* **Field**, a set **F** with two operations (addition and multiplication) satisfying certain axioms: commutativity, associativity, distributivity, additive identity, multiplicative identity, additive inverse, and multiplicative inverse.
* **Vector space**, a set **V** on a field **F** along with an addition and a scalar multiplication on **V**, satisfying certain axioms: commutativity, associativity, additive identity, additive inverse, multiplicative identity, and distributivity.
* **Subspace**, a subset of **V** that is also a vector space.
* **Sum of subspaces**, the smallest subspace containing all the given subspaces.

  $$U + W = \{u + w | u \in U, w \in W\}$$

* **Direct sum**, a sum of subspaces where each element can be uniquely written as a sum of elements from each subspace.

  $$V = U \oplus W \space if \space V = U + W \text{ and } U \cap W = \{0\}$$

* **Span**, the set of all linear combinations of a given set of vectors.

  $$Span(V) = \{a_1v_1 + a_2v_2 + ... + a_nv_n | a_i \in F, v_i \in V\}$$

* **Linear independence**, a set of vectors where no vector can be written as a linear combination of the others.

  $$a_1v_1 + a_2v_2 + ... + a_nv_n = 0 \implies a_1 = a_2 = ... = a_n = 0$$

* **Basis**, a linearly independent set of vectors that spans the entire vector space.
* **Dimension**, the number of vectors in a basis of the vector space.
* **Linear map (linear transformation)**, a function T: **V** → **W** that preserves vector addition and scalar multiplication.

  $$T(u + v) = T(u) + T(v)$$

  $$T(cv) = cT(v)$$
* **Null space (kernel)**, the set of vectors in **V** that map to the zero vector in **W**.

  $$Null(T) = \{v \in V | T(v) = 0\}$$
* **Range (image)**, the set of vectors in **W** that are images of vectors in **V**.

  $$Range(T) = \{T(v) | v \in V\}$$
* **Injective (one-to-one)**, a linear map where different inputs map to different outputs.

  $$T(u) = T(v) \implies u = v$$
* **Surjective (onto)**, a linear map where every element in **W** is the image of at least one element in **V**.

* **Matrix of a linear map**, a way to represent a linear map **T** with respect to chosen bases $v_1, v_2, \ldots, v_n$ and $w_1, w_2, \ldots, w_m$ for **V** and **W** respectively. The matrix **A** is defined such that:

  $$T(v_j) = a_{1j}w_1 + a_{2j}w_2 + ... + a_{mj}w_m$$

  where the coefficients $a_{ij}$ form the columns of the matrix **A**, also denoted as **M(T, (v_1, ..., v_n), (w_1, ..., w_m))** or simply **M(T)** when the bases are understood.

* **Composition of linear maps**, combining two linear maps **S**: **U** → **V** and **T**: **V** → **W** to form a new linear map **T∘S**: **U** → **W**.

  $$(T \circ S)(u) = T(S(u))$$
* **Matrix multiplication**, the operation of multiplying two matrices **A** and **B** to get a new matrix **C**, where the element $c_{ij}$ is computed as:

  $$c_{ij} = \sum_{k} a_{ik}b_{kj}$$

  This definition is deliberately chosen to correspond to the composition of linear maps, so that $M(T \circ S) = M(T)M(S)$.

* **Rank of a matrix**, is the column rank of the matrix, which is the dimension of the span of the columns of the matrix.

* **Invertible**, a linear map **T**: **V** → **W** is invertible if there exists a linear map **S**: **W** → **V** such that:

  $$S(T(v)) = v \space for \space all \space v \in V$$

  $$T(S(w)) = w \space for \space all \space w \in W$$

  In this case, **S** is called the inverse of **T**, denoted as **T**^(-1).

* **Isomorphism**, is an invertible or bijective linear map. If there exists an isomorphism between **V** and **W**, they are said to be isomorphic.