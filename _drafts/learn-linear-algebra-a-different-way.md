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
  - Two finite-dimensional vector spaces over the same field are isomorphic if and only if they have the same dimension.
  - Every finite-dimensional vector space over a field **F** is isomorphic to **F**^n for some non-negative integer n.
  - $L(V, W)$ and $F^{m, \space n}$ are isomorphic if **V** and **W** are finite-dimensional with dimensions n and m respectively.

> 📝 Notes
>
> Here, a higher level of abstraction is introduced by taking **$L(V, W)$**, the set of all linear maps from **V** to **W**, as a vector space itself, which the elements in it are linear maps. This is a **key insight** in linear algebra, allowing us to study the properties of linear maps in a more general and abstract way.
>
> A basis for **L(V,W)** is a set of simple, fundamental linear maps. Each basis map is constructed to do one very specific job: it maps a single basis vector of **V** to a single basis vector of **W** and sends all other basis vectors of **V** to zero.

* **Notion of $v + U$**, where **v** is a vector in **V** and **U** is a subspace of **V**, defined as:

  $$v + U = \{v + u | u \in U\}$$

  This set is called an affine subset or a coset of **U** in **V**. It represents all vectors that can be obtained by adding the vector **v** to each vector in the subspace **U**. Note that **v + U** is not a subspace unless **v** is the zero vector, but it is a translation of the subspace **U** by the vector **v**.

* **Quotient space**, denoted as **V/U**, is the set of all cosets of a subspace **U** in a vector space **V**. Each element of the quotient space is of the form **v + U**, where **v** is a vector in **V**. The quotient space itself can be given a vector space structure by defining addition and scalar multiplication as follows:

  - Addition: $$(v + U) + (w + U) = (v + w) + U$$
  - Scalar multiplication: $$c(v + U) = (cv) + U$$

  for all vectors **v, w** in **V** and all scalars **c** in the field over which the vector space is defined.

  The dimension of the quotient space **V/U** is given by:

  $$dim(V/U) = dim(V) - dim(U)$$

  This means that the dimension of the quotient space is equal to the dimension of the original vector space minus the dimension of the subspace **U**.

  And quotient spaces allow us to study the structure of vector spaces by examining the relationships between subspaces and their complements.

> 📝 Notes
> This is another case of abstraction in linear algebra, where we deliberately construct a new vector space (the quotient space) from an existing vector space and one of its subspaces. And its element is not individual vectors, but rather a set of vectors - called cosets or a translate of the subspace in question.

* **Linear functional**, is a linear map from **V** to **F**. In other words, a linear functional is an element of **L(V, F)**.

* **Dual space**, denoted as **V'**, is the vector space of all linear functionals on **V**. **V' = L(V, F)**. Suppose **V** is finite-dimensional with dimension n, then **V'** is also finite-dimensional with the same dimension n.

* **Dual basis**, if $v_1, v_2, \ldots, v_n$ is a basis for **V**, then the dual basis for **V'** is a set of linear functionals $φ_1, φ_2, \ldots, φ_n$ such that:

  $$φ_i(v_j) = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

  for all i, j from 1 to n.

> 📝 Notes
>
> This is a special case of **L(V, W)** where **W** is the field **F**.

* **Dual map**, suppose **T**: **V** → **W** is a linear map, then the dual map **T'**: **W'** → **V'** is defined by:

  $$(T'(φ))(v) = φ(T(v))$$

  for all φ in **W'** and all v in **V**.

  In other words, the dual map **T'** takes a linear functional φ on **W** and produces a linear functional on **V** by pre-composing φ with the linear map **T**.

* **Annihilator**, for $U \subseteq V$, the annihilator of **U**, denoted as $$U^0$$, is defined as:

  $$U^0 = \{φ \in V' | φ(u) = 0 \text{ for all } u \in U\}$$

  In other words, the annihilator $$U^0$$ is the set of all linear functionals on **V** that vanish on every vector in the subspace **U**.

  $$U^0$$ is a subspace of **V'**, and if **V** is finite-dimensional, then:

  $$dim(U) + dim(U^0) = dim(V)$$

> 📝 Notes
>
> The dimension formula above is really interesting, that it expresses a relationship between the dimension of subspaces on field **F**, and the dimension of a subspace on field of linear functionals.
>
> Intuitively, we can say that every degree of freedom in **V** is either captured by movement inside **U**, or by a constraint (linear functionals) in **U^0**.
>
> We see there are 3 different vector spaces involved here:
  * The original vector space **V** and its subspace **U**, on field **F**.
  * The dual space **V'** and its subspace $U^0$, on field of **linear functionals**.
  * All dual maps **T'**: **W'** → **V'**, on field of **linear map which maps linear functionals to linear functionals.**
