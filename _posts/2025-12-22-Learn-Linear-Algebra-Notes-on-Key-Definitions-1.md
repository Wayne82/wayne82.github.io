---
layout: post
title:  "Learn Linear Algebra a Different Way - Notes on Key Definitions (1)"
categories: mathematics
mathjax: true
comments: true
---

Continuing from what I've left off in my [previous post](https://wayne82.github.io/mathematics/2025/09/20/Learn-Linear-Algebra-Notes-on-Key-Definitions.html), I will proceed with my learning notes on key definitions in the remaining chapters of [Linear Algebra Done Right](https://linear.axler.net/LADR4e.pdf) by Sheldon Axler.

## The Key Definitions (Chapter 4~6)
* **Polynomial**, A polynomial with coefficients in (F) is a formal expression of the form

  $$ p(z) = a_0 + a_1 z + a_2 z^2 + ... + a_m z^m $$

  where (m) is a nonnegative integer and (a_0, a_1, ..., a_m) are elements of (F) with (a_m \neq 0). The set of all polynomials with coefficients in (F) is denoted by (P(F)).

* **Fundamentional Theorem of Algebra (First Version)**, Every nonconstant polynomial with complex coefficients has at least one complex root.

* **Fundamentional Theorem of Algebra (Second Version)**, Every nonconstant polynomial with complex coefficients has exactly (m) complex roots, counted with multiplicities.

* **Factorization of Polynomials over R**, Every nonconstant polynomial with real coefficients can be factored as a product of real polynomials of degree at most 2.

  $$ p(x) = c(x - \lambda_1)...(x - \lambda_m)(x^2 + b_1*x + c_1) ... (x^2 + b_M*x + c_M) $$

  where (c, $\lambda_1, ..., \lambda_m, b_1, ..., b_M, c_1, ..., c_M$) are real numbers, with $b_i^2 - 4c_i < 0$ for each $i = 1, 2, ..., M$.

* **Operator**, A linear map from a vector space to itself is called an operator.

* **Invariant Subspace**, Suppose $T \in L(V)$. A subspace $U$ of $V$ is called invariant under $T$ if $T(u) \in U$ for every $u \in U$.

  > 📝 Note
  >
  > We could consider there are 3 different kinds of subspaces:
  > * **Random Subspace**: Vectors in the subspace get kicked out ($T(u) \notin U$).
  > * **Invariant Subspace**: Vectors in the subspace stay in the subspace ($T(u) \in U$), but might be rotated, sheared, or mixed.
  > * **Eigenspace**: Vectors in the subspace stay in the subspace and are only scaled ($T(u) = \lambda u$ for some scalar $\lambda$).

* **Eigenvalue and Eigenvector**, Suppose $T \in L(V)$. A scalar $\lambda$ is called an eigenvalue of $T$ if there exists a nonzero vector $v \in V$ such that $T(v) = \lambda v$. The vector $v$ is called an eigenvector of $T$ corresponding to the eigenvalue $\lambda$.

* **Linear Independent Eigenvectors**, Suppose $T \in L(V)$. Then every list of eigenvectors of $T$ corresponding to distinct eigenvalues of $T$ is linearly independent.

* **Notation: $p(T)$**, Suppose $T \in L(V)$ and $p \in P(F)$, then $p(T)$ is the operator on $V$ defined by

  $$ p(T) = a_0 I + a_1 T + a_2 T^2 + ... + a_m T^m $$

  where $I$ is the identity operator on $V$.

* **Null Space and Range of $p(T)$ are invariant under $T$**, Suppose $T \in L(V)$ and $p \in P(F)$. Then the null space and range of $p(T)$ are invariant under $T$.

  > 📝 Notes
  >
  > Null space and Range of $T$ are also invariant under $T$.

* **Existence of Eigenvalues**, Every operator on a finite-dimensional complex vector space has an eigenvalue

* **Minimal Polynomial**, The minimal polynomial of an operator $T \in L(V)$ is the **unique** monic polynomial $m_T \in P(F)$ of least degree such that $m_T(T) = 0$.

  > 📝 Note
  >
  > The essence of the minimal polynomial is that it represents the most efficient way to **annihilate** the operator $T$, or turn it into the **zero operator**.
  >
  > The proof of the existence of the minimal polynomial can be constructed this way,
  > 1. Pick a non-zero $v \in V$.
  > 2. Construct a list of vectors $v, T(v), T^2(v), ..., T^m(v)$ until we reach a point where the list becomes linearly dependent.
  > 3. The linear dependence relation gives us a monic polynomial $q \in P(F)$ such that $q(T)(v) = 0$. So, $v$ is in the null space of $q(T)$.
  > 4. Then, we just construct a polynomial $q$ such that $q(T)$ annihilates the null space of $q(T)$.
  > 5. Vector space $V$ can be decomposed into null q(T) and range q(T).
  > 6. Apply the same process to the subspace of range $q(T)$, and repeat until we reach the zero dimension. Each repetition gives us a new monic polynomial.
  > 7. At the end, the minimal polynomial is the product of all these monic polynomials.
  >
  > The repetition process is equivalent to use induction method, as the official proof is described in the book.
  >

* **Eigenvalues are the zeros of the minimal polynomial**, The eigenvalues of an operator $T \in L(V)$ are exactly the zeros of the minimal polynomial $m_T$.

* **Operators on Odd-Dimensional Spaces**, Every operator on an odd-dimensional **real** vector space has an eigenvalue.

  > 📝 Note
  >
  > * We already know that every operator on a finite-dimensional complex vector space has an eigenvalue, so this is specific for real vector spaces.
  > * The proof of this lemma relies on another lemma, "Suppose $T \in L(V)$ and $b, c \in R$ with $b^2 - 4c < 0$. Then dim null($T^2 + bT + cI$) is an even number."

* **Upper Triangular Matrix**, A square matrix is called upper triangular if all entries below the diagonal are zero.

* **Eigenvalues of Upper Triangular Matrices**, The eigenvalues of an upper triangular matrix are exactly the entries on its diagonal.

* **Condition to Have an Upper Triangular Matrix**, T has an upper triangular matrix with respect to some basis of $V$ if and only if the minimal polynomial of $T$ equals $(z - \lambda_1)...(z - \lambda_m)$ for some $\lambda_1, ..., \lambda_m \in F$.

  > 📝 Note
  >
  > Immediately, we can get every operator on a finite-dimensional complex vector space has an upper triangular matrix with respect to some basis of $V$.

* **Diagonal Matrix**, A diagonal matrix is a square matrix that is 0 everywhere except possibly on the diagonal.

* **Diagonalizable Operator**, An operator $T \in L(V)$ is called diagonalizable if the operator has a diagonal matrix with respect to some basis of $V$.

* **Conditions Equivalent to Diagonalizability**, Let $\lambda_1, ..., \lambda_m$ be the distinct eigenvalues of $T$. The following conditions are equivalent for an operator $T \in L(V)$:
  * $T$ is diagonalizable.
  * $V$ has a basis consisting of eigenvectors of $T$.
  * $V = E(\lambda_1, T) \oplus \cdots \oplus E(\lambda_m, T)$, where $E(\lambda_i, T)$ is the eigenspace of $T$ corresponding to $\lambda_i$.
  * $dimV = dimE(\lambda_1, T) + ... + dimE(\lambda_m, T)$.

  Then, if T has dimV distinct eigenvalues, then T is diagonalizable.

* **Necessary and Sufficient Conditions for Diagonalizability**, T is diagonalizable if and only if the minimal polynomial of T equals $(z - \lambda_1)...(z - \lambda_m)$ for some list of distinct $\lambda_1, ..., \lambda_m \in F$.

  > 📝 Note
  >
  > The difference of the above condition and the condition to have an upper triangular matrix is that the minimal polynomial of T must have distinct roots.

* **Gershgorin Disk Theorem**, Each eigenvalue of T is contained in some Gershgorin disk of T, where the Gershgorin disks are defined as follows:

  $$ D_i = \{ z \in F : |z - A_{j,j}| \leq \sum_{k = 1, j \neq k}^{n} |A_{j, k}| \} $$

  for each $j = 1, 2, ..., n$, where $A_{j, k}$ is the entry in the $j$-th row and $k$-th column of the matrix representation of T.

* **Commute**, Two operators $S, T \in L(V)$ commute if $S(T(v)) = T(S(v))$ for all $v \in V$. And S and T commute if and only if their matrix representations with respect to some basis of $V$ commute.
  * **Eigenspace is Invariant Under Commuting Operators**, If $S, T \in L(V)$ commute, then the eigenspace of $S$ corresponding to an eigenvalue $\lambda$ is invariant under $T$.
  * **Simultaneous Diagonalizability**, Two diagonalizable operators on the same vector space have diagonal matrices with respect to the same basis if and only if they commute.
  * **Common Eigenvector for Commuting Operators**, Every pair of commuting operators on a finite-dimensional nonzero complex vector space has a common eigenvector.

* **Commuting Operators are Simultaneously Upper Triangularizable**, Two commuting operators on a finite-dimensional complex vector space have upper triangular matrices with respect to the same basis.

* **Inner Product**, An inner product on a vector space $V$ over a field $F$ is a function that takes each ordered pair (u, v) of vectors in $V$ and returns a scalar in $F$, denoted by $\langle u, v \rangle$, satisfying the following properties:
  1. **Positivity**: $\langle u, u \rangle \geq 0$ for all $u \in V$.
  2. **Definiteness**: $\langle u, u \rangle = 0$ if and only if $u = 0$.
  3. **Additivity in first slot**: $\langle u + v, w \rangle = \langle u, w \rangle + \langle v, w \rangle$ for all $u, v, w \in V$.
  4. **Homogeneity in first slot**: $\langle cu, v \rangle = c \langle u, v \rangle$ for all $u, v \in V$ and $c \in F$.
  5. **Conjugate symmetry**: $\langle u, v \rangle = \overline{\langle v, u \rangle}$ for all $u, v \in V$.
* **Inner Product Space**, A vector space $V$ over a field $F$ with an inner product is called an inner product space.

* **Norm**, The norm of a vector $v \in V$ is defined as $$\|v\| = \sqrt{\langle v, v \rangle}$$.

* **Orthogonal**, Two vectors $u, v \in V$ are orthogonal if $\langle u, v \rangle = 0$.

* **Pythagorean Theorem**, For any two vectors $u, v \in V$, we have $$\|u + v\|^2 = \|u\|^2 + \|v\|^2$$ if and only if $u$ and $v$ are orthogonal.

* **Cauchy-Schwarz Inequality**, For any two vectors $u, v \in V$, we have $$\|\langle u, v \rangle\| \leq \|u\| \|v\|$$. The equality holds if and only if $u$ and $v$ are linearly dependent.

  > 📝 Note
  >
  > The Cauchy-Schwarz inequality is arguably the most important inequality in all of linear algebra. Without this inequality, we couldn't define the angle in inner product space. Because the angle is defined as $\cos \theta = \frac{\langle u, v \rangle}{\|u\| \|v\|}$, and the Cauchy-Schwarz inequality guarantees that the absolute value of the cosine is always less than or equal to 1.

* **Triangle Inequality**, For any two vectors $u, v \in V$, we have $$\|u + v\| \leq \|u\| + \|v\|$$. The equality holds if and only if $u$ and $v$ are linearly dependent.

* **Orthonormal**, A list of vectors is called orthonormal if each vector in the list has norm 1 and is orthogonal to every other vector in the list.
  $$\langle e_i, e_j \rangle = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$
  * $$\|a_1 e_1 + \cdots + a_n e_n\|^2 = |a_1|^2 + \cdots + |a_n|^2$$
  * Every orthonormal list of vectors is linearly independent.
  * **Bessel's Inequality**, For any vector $v \in V$ and any orthonormal list of vectors $e_1, ..., e_n$, we have $$\|v\|^2 \geq \|\langle v, e_1 \rangle\|^2 + \cdots + \|\langle v, e_n \rangle\|^2$$

* **Orthonormal Basis**, An orthonormal basis of an inner product space $V$ is an orthonormal list of vectors that spans $V$. Suppose $e_1, ..., e_n$ is an orthonormal basis of $V$,
  * $v = \langle v, e_1 \rangle e_1 + \cdots + \langle v, e_n \rangle e_n$
  * $\|v\|^2 = \|\langle v, e_1 \rangle\|^2 + \cdots + \|\langle v, e_n \rangle\|^2$
  * $\langle u, v \rangle = \langle u, e_1 \rangle \overline{\langle v, e_1 \rangle} + \cdots + \langle u, e_n \rangle \overline{\langle v, e_n \rangle}$
  * Every finite-dimensional inner product space has an orthonormal basis.
  * Every orthonormal list exteds to an orthonormal basis of $V$.

* **Gram-Schmidt Process**, The Gram-Schmidt process is a method for constructing an orthonormal basis from a basis of a vector space. Given a basis $v_1, ..., v_n$ of $V$, the Gram-Schmidt process produces an orthonormal basis $e_1, ..., e_n$ as follows:
  1. Let $f_1 = v_1$.
  2. For $i = 2, ..., n$, let
     $$ f_i = v_i - \frac{\langle v_i, f_1 \rangle}{\|f_1\|^2} f_1 - \cdots - \frac{\langle v_i, f_{i-1} \rangle}{\|f_{i-1}\|^2} f_{i-1} $$
  3. Normalize $f_i$ to get $e_i = \frac{f_i}{\|f_i\|}$.

* **Schur's Theorem**, Every operator $T \in L(V)$ on a finite-dimensional complex vector space has an upper triangular matrix with respect to some orthonormal basis of $V$.

* **Riesz Representation Theorem**, Every continuous linear functional on a finite-dimensional inner product space can be represented as an inner product with a fixed vector in the space. Specifically, for every linear functional $f$ on an inner product space $V$, there exists a unique vector $v \in V$ such that for all $u \in V$, we have $$ f(u) = \langle u, v \rangle $$.

  > 📝 Note
  >
  > This is a very surprising result, as it shows that every linear functional can be represented as an inner product with a fixed vector in the space.
  > * It does not care what elements are in the vector space, as long as they follow the rules of a **vector space**.
  > * It also does not care how the functional is defined, as long as it is **linear**.
  > * It also does not care which inner product is used, as long as it satisfies the properties of an inner product.
  >
  > **Crucial Nuance**: the specific vector $v$ changes depending on the inner product we choose.

* **Orthogonal Complement**, The orthogonal complement of a subspace $U$ of an inner product space $V$ is the set of all vectors in $V$ that are orthogonal to every vector in $U$. It is denoted by $U^\perp$ and defined as follows:

  $$ U^\perp = \{ v \in V : \langle u, v \rangle = 0 \text{ for all } u \in U \} $$

* **Orthogonal Projection**, The orthogonal projection of $V$ onto $U$ is the operator $P_U \in L(V)$ defined as follows: For each $v \in V$, write $v = u + w$, where $u \in U$ and $w \in U^\perp$. Then, the orthogonal projection of $v$ onto $U$ is given by $P_U(v) = u$.

* **Minimizing Distance**, The orthogonal projection of $v \in V$ onto $U$ minimizes the distance from $v$ to $U$. Specifically, for any $u \in U$, we have

  $$ \|v - P_U(v)\| \leq \|v - u\| $$

* **Pseudoinverse**, Suppose that $V$ is finite dimensional and $T \in L(V, W)$. The pseudoinverse $T^+ \in L(W, V)$ of $T$ is the linear map from W to V defined by

  $$ T^+(w) = (T|_{null(T)^\perp})^{-1} P_{range(T)}(w) $$

  for each $w \in W$.

* **Pesudoinverse Provides Best Approximation**, Suppose $V$ is finite dimensional, and $T \in L(V, W)$, and $b \in W$. Then,
  * If $x \in V$, $$\|T(T^+b) - b\| \leq \|Tx - b\|$$
  * If $x \in T^+b + null(T)$, then $$\|T^+b\| \leq \|x\|$$

## Conclusion
These chapters introduce numerous foundational definitions and lemmas. The Fundamental Theorem of Algebra serves as a cornerstone for richer theories in linear algebra, particularly through the application of polynomials to operators. A key insight is that every operator possesses a minimal polynomial, the roots of which correspond exactly to the operator's eigenvalues.

Eigenvalues and eigenvectors act as the structural 'building blocks' of an operator and are essential for deciphering its internal mechanics. While operators in complex vector spaces can always be represented by an upper-triangular matrix, they are diagonalizable if and only if their minimal polynomial has distinct roots. Consequently, a vector space can be decomposed into a direct sum of eigenspaces only when the operator is diagonalizable.

Finally, inner product spaces introduce the vector space with geometric structure by defining length and angle. A very surprising result is the Riesz Representation Theorem, which shows that every linear functional can be uniquely represented as an inner product with a specific vector in the space.

The learning notes of the rest of the 3 chapters will go on in the future post.