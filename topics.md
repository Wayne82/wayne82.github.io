---
layout: default
title: "Topics"
description: "Organize by Topics"
permalink: /topics/
---

<header class="post-header">
    <h1 class="post-title p-name" itemprop="name headline">{{ page.description | escape }}</h1>
</header>

<div class="topics-grid">
    {% assign topics = site.topics | sort: 'order' %}
    {% for topic in topics %}
        {% assign category_posts = site.posts | where: "categories", topic.category %}
        <div class="topic-card">
            <a href="{{ topic.url | relative_url }}" class="topic-card-link">
                <div class="topic-card-content">
                    <h2 class="topic-card-title">{{ topic.title }}</h2>
                    <p class="topic-card-description">{{ topic.description }}</p>
                    <div class="topic-card-meta">
                        <span class="post-count">{{ category_posts.size }} post{% if category_posts.size != 1 %}s{% endif %}</span>
                    </div>
                </div>
            </a>
        </div>
    {% endfor %}
</div>

<style>
.topics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 30px;
}

.topic-card {
    border: 1px solid #e1e4e8;
    border-radius: 8px;
    overflow: hidden;
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}

.topic-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.topic-card-link {
    display: block;
    text-decoration: none;
    color: inherit;
    height: 100%;
}

.topic-card-link:hover {
    text-decoration: none;
}

.topic-card-link:hover .topic-card-title {
    text-decoration: underline;
    color: black;
}

.topic-card-content {
    padding: 20px;
    display: flex;
    flex-direction: column;
    min-height: 200px;
}

.topic-card-title {
    margin: 0 0 10px 0;
    font-size: 1.4em;
    font-weight: 600;
    color: #2a7ae4;
}

.topic-card-description {
    color: #666;
    line-height: 1.6;
    margin-bottom: 15px;
    flex-grow: 1;
}

.topic-card-meta {
    margin-top: auto;
    padding-top: 15px;
    border-top: 1px solid #eee;
}

.post-count {
    color: #999;
    font-size: 0.9em;
    font-weight: 500;
}

@media (max-width: 768px) {
    .topics-grid {
        grid-template-columns: 1fr;
        gap: 15px;
    }
    
    .topic-card-content {
        padding: 15px;
    }
}
</style> 