---
title: "Projects"
date: 2022-08-21T21:50:54-07:00
tags:
  - technical
  - evergreen
---

The list below are notable projects I'm currently maintaining/finished. A more incomplete list of ideas that I will do sometime can be found in [[cache/backburners|backlog]].

## BentoML -- Unified Model Serving Framework 🍱

BentoML is a framework that simplifies ML model deployment and provides a faster way to ship your ML model to production.

We recently rewrote the library for our 1.0 releases, including a new design to improve serving performance, provide a new packaging format for machine learning application, and easy integration with SOTA ML frameworks natively.

The container generation features support [OCI-compliant](/cache/docker.md) container, where we provides multiple architecture support, GPU support, automatic generation upon build time, with efficient caching implemented to reduce build time and improve agility.

I have also implement gRPC support for a BentoServer, enable better interoperability between existing Kubernetes infrastructure where gRPC is used and newly created Bento.

_Built with: Python, Jinja, Go, BuildKit, gRPC_

## onw -- A real-time navigation tools for safer commute 📌

[onw](https://github.com/tiproad/omw) is a real-time navigation tool that enables users to safely commute to their destination with greater peace of mind. We implemented features such as route optimization, heat map visualization to indentifies hot zones, peer notification system.

I implemented a Gaussian Mixture Model to find the safest path between different locations, trained on past assault data provided by Toronto Police Department. I then use Google Maps API to implements hot zones from given prediction results, then shipped to a React Native app using Expo and AWS Fargate.

Awarded: Finalists at [Hack the North 2021](https://devpost.com/software/twogether).

_Built with: AWS Fargate, React Native, TypeScript, GraphQL, Apache Spark MLlib, Google Maps API_

## dha-ps -- Price Recommender System 📈

I designed and implemented from scratch a product-based price recommender system.
Performed transfer learning on a BERT model for similarity recommender on users'
provided datasets, and then serve with FastAPI, with 93% code coverage.

Built a dockerized Go microservices and deployed to Google Kubernetes Engine (GKE).
Currently running in production with serving 15RPS for each recommender query.

_Built with: Go, Gorrila Mux, PyTorch, HuggingFace, GKE_
