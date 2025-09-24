---
layout: post
title:  "Learn Linear Algebra a Different Way - Notes on Linear Map Lemma"
categories: mathematics
mathjax: true
comments: true
---

Linear Map Lemma is a fundamental result in linear algebra that essentially says a linear map is completely and uniquely determined by what it does to a set of basis vectors. This lemma is crucial because it allows us to understand and work with linear maps in a more manageable way by focusing on their action on a basis. The first time I read this lemma, I didn't quite reason about it in a way that made sense to me, because I am thinking the linear map is a function, and functions are defined by a kind of equation. When the lemma said the linear map can be uniquely defined by its action on a basis, I was puzzled by imagining how a fixed function could be determined this way. Eventually, I figured out it is not about the function itself, but rather about the transformation from the domain vector space to the codomain vector space. In this blog, I'd like to share my understanding on this.

## Linear Map Lemma Statement
Let **V** and **W** be vector spaces over the same field **F**, and let **T**: **V** → **W** be a linear map. If **{v_1, v_2, ..., v_n}** is a basis for **V**, then the action of **T** on the basis vectors completely determines **T**. Specifically, if we know **T(v_i)** for all **i**, we can uniquely extend **T** to the entire space **V**.