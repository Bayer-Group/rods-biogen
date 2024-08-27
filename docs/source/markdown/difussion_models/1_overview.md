# Quickstart

This document gives an overall quick introduction on latent diffusion models: How they work,
what they are capable of, the main concepts needed to understand them, and their applications to
biomedical image tasks.

## Introduction

### Latent diffusion

This section is based on the {ref}`stable diffusion blog post from huggingface<stable_diffusion_hfpost>`
Generally speaking, diffusion models are machine learning systems that are trained to denoise random Gaussian noise step by step, to get to a sample of interest, such as an image.

:::{figure-md} fig-target
:class: myclass

<img src="../images/diffusion_figure.png" alt="diffusion" class="bg-primary mb-1" width="600px">

Example of a diffusion process. At each step, the network predicts how to denoise the current image.
:::

Latent diffusion can reduce the memory and compute complexity by applying the diffusion process over a lower dimensional latent space, instead of using the actual pixel space. This is the key difference between standard diffusion and latent diffusion models: in latent diffusion the model is trained to generate latent (compressed) representations of the images.

There are four main components in latent diffusion.

1. An autoencoder (VAE) that compresses the input images into a latent space.
2. A U-Net that learns how to reverse the diffusion process so we can generate images from gaussian noise.
3. A text-encoder, e.g. CLIP's Text Encoder to create context embeddings used to guide the image generation process.
4. A noise scheduler that defines the diffusion process to transform noise into images.

### What is stable diffusion?

Stable Diffusion is a text-to-image latent diffusion model created by the researchers and engineers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/) and [LAION](https://laion.ai/). It is trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database. LAION-5B is the largest, freely accessible multi-modal dataset that currently exists.

* Short description of the topic assuming the reader has no prior knowledge of the topic
* Estimated time to understand the topic
* Estimated level of difficulty (Introductory, Easy, Medium, Advanced, Pro)
* Brief description of the core concepts explained in the current topic

## TLDR

* Super quick summary of the contents of `topic.md`.

## Goals and use cases

* Description of why the topic is useful
* Description of what you can achieve when understanding the topic

## Pros, Cons & alternatives

* Description of the main benefits of the topic
* Description of the main drawbacks of the topic
* Trade-offs & implications of using the topic

## Requirements and background

* Recommended prerequisites to understand the topic and a classification of its difficulty.
* Tools used to utilize and understand the topic.  
* Other related topics
* Alternatives to the topic, if any.

## Learning

* Advice on how to learn about the topic.
* Description of external resources to learn about the topic.
* Introductory tutorials

## Reference

* Advice on resources that come in handy when using the topic for people familiar with it.
