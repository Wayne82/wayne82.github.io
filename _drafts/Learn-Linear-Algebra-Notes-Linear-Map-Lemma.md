---
layout: post
title:  "Learn Linear Algebra a Different Way - Notes on Linear Map Lemma"
categories: mathematics
mathjax: true
comments: true
---

Linear Map Lemma is a fundamental result in linear algebra that essentially says a linear map is completely and uniquely determined by what it does to a set of basis vectors. This lemma is crucial because it allows us to understand and work with linear maps in a more manageable way by focusing on their action on a basis. The first time I read this lemma, I didn't quite reason about it in a way that made sense to me, because I am thinking the linear map is a function, and functions are defined by a kind of equation. When the lemma said the linear map can be uniquely defined by its action on a basis, I was puzzled by imagining how a fixed function could be determined this way. Eventually, I figured out it is not about the function itself, but rather about the transformation from the domain vector space to the codomain vector space. In this blog, I'd like to share my understanding on this.

## The Statement of the Lemma
Let ${v_1, v_2, ..., v_n}$ be a basis of a vector space **V**, and let ${w_1, w_2, ..., w_n}$ be any vectors in another vector space **W**.

The lemma states that there exists a **unique linear map** **T**: **V** → **W** such that:

$$T(v_i) = w_i$$

for each $$i = 1, 2, ..., n$$.

In simpler terms: if you have a basis for your starting space (**V**), you can define a linear map by simply choosing where each basis vector goes. Once you've made that choice, the entire map is set in stone, and there's only one possible linear map that satisfies your choices.

## The Initial Confusion
The initial confusion is what I mentioned in the first paragraph: I was thinking of a linear map as a function defined by an equation. When the lemma said the linear map can be uniquely defined by its action on a basis, I was puzzled by imagining how a fixed function could be determined this way.

This leads to the very heart of the question: what a function or linear map is in mathematics. I asked my confusion to Google Gemini, and I get a very good response, which I will summarize right in the next section.

## Understanding What is Linear Map
### What It Means for Two Linear Maps to Be Equal
Following the principle of **extensionality** in mathematics, two functions (or linear maps) are considered identical if and only if they meet two conditions:
* They have the **exact same domain**.
* For every single element in that domain, they produce the **exact same output**.

So, if we have two linear maps, $S: V \to W$ and $T: V \to W$, we say $S$ and $T$ are equal if and only if $S(v) = T(v)$ for **every vector** $v \in V$.

> 📝 Notes
>
> In mathematics, **extensionality** is the principle that an object's identity is determined solely by **its contents**, not by **how it is defined or described**. The most common example is the axiom of extensionality in set theory, which states that two sets are equal if and only if they have exactly the same members. This means that regardless of the words used to define a set or the method by which its elements were gathered, two sets with the identical collection of elements are considered the same set.

### "Form" vs "Function"
We don't care about the "form" of the linear map. The linear map is defined by its input-output relationship, not the specific formula we write down.

For example, in regular algebra, the functions $f(x)=(x−1)(x+1)$ and $g(x)=x^2−1$ are the **exact same function**. They might look different, but for every input $x$, they produce the same output.

Similarly, I could define a linear map $T:R^2 \to R^2$ by the formula $T(x,y)=(x+y,x−y)$. Someone else could define a map $S$ by specifying its action on the basis vectors: $S(1,0)=(1,1)$ and $S(0,1)=(1,−1)$. The **Linear Map Lemma** shows that these are not just two different maps that happen to give the same answers; they are the same map. Their "form" or "definition" started differently, but their action on the vector space is identical.

### The Domain is the Vector Space
This is another key point. When we define a linear map $T:V \to W$:
* The domain is exactly the **vector space $V$**. It's not some larger space that happens to include $V$.
* The map is only defined for elements within $V$. The question of what $T$ might do to something outside of $V$ is meaningless because the domain is precisely $V$.

This is why we care so much about properties like closure in vector spaces — it guarantees that our operations stay within the defined domain.

### Connects Back to the Lemma
To prove that two linear maps S and T are the same, we could check every single vector $$v \in V$$ and see if $$S(v)=T(v)$$. But if $$V$$ is anything other than the zero space, it contains infinitely many vectors! That's an impossible task.

The lemma tells us: "Don't bother checking all infinite vectors. Just check the handful of vectors in a basis."

If $$v_1, v_2, \ldots, v_n$$ is a basis for $$V$$, and you can show that:

$$S(v_i) = T(v_i)$$

for each $$i = 1, 2, \ldots, n$$, then the lemma's guarantee of **uniqueness** allows you to conclude that $$S = T$$ for the entire space. You're done.

I will omit the proof here, as the book "[Linear Algebra Done Right](https://linear.axler.net/LADR4e.pdf)" by Sheldon Axler has a very clear proof of it. And once the confusion is resolved, the proof is clear and straightforward.

## The Basis of L(V, W)
Let's take a look at an application of the lemma.

As we know, all linear maps from a vector space **V** to a vector space **W** form a **vector space** themselves, denoted as **L(V, W)**. The "vectors" in this space are the linear maps. Then, what is the basis of this vector space?
To find it, we can use the **Linear Map Lemma**.

The **Linear Map Lemma** tells us that to define a unique linear map, we only need to decide where the basis vectors of the domain V are sent. We can use this principle to construct the simplest possible non-zero linear maps, which will serve as the basis for the entire space $L(V,W)$.

Suppose:
* **V** has dimension $$n$$ with basis $${v_1, v_2, \ldots, v_n}$$.
* **W** has dimension $$m$$ with basis $${w_1, w_2, \ldots, w_m}$$.

To build a basis "vector" for the space $L(V,W)$, we need to define a single linear map. Let's call one such map $T_{ij}$. According to the lemma, we just have to say what $T_{ij}$ does to each $v_k$ in the basis of $V$.

We can define $T_{ij}$ as follows:

$$
T_{ij}(v_k) = \begin{cases}
w_j & \text{if } k = i \\
0 & \text{if } k \neq i
\end{cases}
$$

This effectively sends the $i$th basis vector of $V$ to the $j$th basis vector of $W$. Send all other basis vectors of $V$ to the zero vector in $W$. And this creates a "building-block" map. It has exactly one job: to connect one specific starting basis vector ($v_i$) to one specific destination basis vector ($w_j$).

Since we can do this for every combination of a starting vector $v_j$ (of which there are $n$) and a destination vector $w_i$ (of which there are $m$), we can create $n \times m$ unique, simple linear maps. This set of all possible $T_{ij}$ maps forms the basis for $L(V, W)$.

Q.E.D.