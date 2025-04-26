---
layout: page
title: Gallery
permalink: /gallery/
---

This is the gallery of My son's art works.

<ul>
  {% assign galleries = site.galleries | sort: 'order' %}
  {% for gallery in galleries %}
    <li>
      <h2><a href="{{ gallery.url }}">{{ gallery.title }}</a></h2>
    </li>
  {% endfor %}
</ul>