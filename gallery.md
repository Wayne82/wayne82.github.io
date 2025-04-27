---
layout: default
title: "Gallery"
description: "Art Works Gallery from My Son"
permalink: /gallery/
---
<header class="post-header">
<h1 class="post-title p-name" itemprop="name headline">{{ page.description | escape }}</h1>
</header>

<div class="gallery-grid">
  {% assign galleries = site.galleries | sort: 'order' %}
  {% for gallery in galleries %}
    {% assign gallery_data = site.data.galleries[gallery.data] %}
    {% assign name = gallery_data.images[gallery_data.cover_image].name %}
    {% assign src = gallery_data.path | append: '/' | append: name %}

    <div class="gallery-card" data-pswp-gallery="gallery-list">
      <a href="{{ gallery.url | relative_url }}" class="gallery-card-link">
        <div class="gallery-card-image-wrapper">
          <img src="{{ src }}" alt="{{ gallery.title }}" class="gallery-card-image" />
        </div>
        <div class="gallery-card-title">{{ gallery.title }}</div>
      </a>
    </div>
  {% endfor %}
</div>

<link rel="stylesheet" href="{{ '/assets/css/gallery.css' | relative_url }}">
<link rel="stylesheet" href="{{ '/assets/css/photoswipe.css' | relative_url }}">

<script type="module">
  import PhotoSwipeLightbox from '/assets/js/photoswipe/photoswipe-lightbox.esm.min.js';
  const lightbox = new PhotoSwipeLightbox({
    gallery: '.gallery-grid',
    children: 'a[data-pswp-src]',
    pswpModule: () => import('/assets/js/photoswipe/photoswipe.esm.min.js'),
  });
  lightbox.init();
</script>