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
### What It Means for Two Maps to Be Equal
