---
layout: post
title:  "Learn Linear Algebra a Different Way - Notes on Key Definitions Cont."
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