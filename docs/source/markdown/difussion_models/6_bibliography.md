# Bibliography

This section contains all external links that have been referenced in the other documents of this template.

## Blog posts on diffusion models

```{eval-rst}
.. [stable_diffusion_hfpost] 
    ---**Stable diffusion overview**
    
    Blog post by HuggingFace that gives an overall description of how stable diffusion works. It is a high-level tutorial that focuses on showcasing the pipelines available in the `diffusers` library, and serves as an starting point for generating images and getting familiar with the library features.    

    https://huggingface.co/blog/stable_diffusion

.. [annotated_diffusion_hfpost] 
   ---**Annotated diffusion**
   
   Blog post by HuggingFace that describes the theoretical foundations of stochastic diffusion models, as well as an step-by-step implementation in Pytorch. It focuses in explaining the match behind diffusion models and the architectures used to implement them in higher detail than the stable diffusion overview.

   https://huggingface.co/blog/annotated-diffusion

.. [generative_models_overview]
   ---**Generative models overview**

   Blog post explaining the different types of generative models, their advantages, disadvantages, and tradeoffs.
   
   https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#:~:text=Compared%20to%20DDPM%2C%20DDIM%20is,have%20similar%20high%2Dlevel%20features

.. [laion_5b_dataset]
   ---**Exploring 12 Million of the 2.3 Billion Images Used to Train Stable Diffusion’s Image Generator**

   This blogpost analyzes the structure and types of images present in a subset of the LAION-5B dataset. It explores 
   what data sources and domains the data was scrapped from, which famous artists are present, famous people and fictional characters present in the dataset, and the NSFW content used to train stable diffusion.

   https://waxy.org/2022/08/exploring-12-million-of-the-images-used-to-train-stable-diffusions-image-generator/

.. [sd_image_compression]
   ---**Stable Diffusion based Image Compression**

   Matthias Buehlmann used the stable diffusion autoencoder to build an image compression system. This blogpost describes
   how the pre-trained VAE workson different types of images and the artifacts it produces.

   https://matthias-buehlmann.medium.com/stable-diffusion-based-image-compresssion-6f1f0a399202

.. [vq_gan_post]
   ---**The Illustrated VQGAN**

   This blog post explains how VQGAN works as well as how it can be used alongside CLIP to generate hight
   quality images.

   https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/
```

## Relevant papers

```{eval-rst}
.. [unet_paper] 
   ---**U-Net: Convolutional Networks for Biomedical Image Segmentation**
   
   This is the paper that first introduced the Unet architecture in the context of medical image segmentation.

   https://arxiv.org/pdf/1505.04597.pdf

.. [latent_diffusion_paper] 
   ---**High-Resolution Image Synthesis with Latent Diffusion Models**
   
   The website of the paper that introduced stochastic diffusion models on latent space (previously it was done directly on image space). It shows a lot of examples of what this models can achieve and provides links to the paper on arxiv, as well as the GitHub repository.

   https://ommer-lab.com/research/latent-diffusion-models/

.. [vae_training]
   --- **Taming transformers for high-resolution image synthesis (A.K.A #VQGAN)**

   *Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. CoRR, abs/2012.09841, 2020. 2, 3, 4, 6, 7, 21, 22, 29, 34, 36*

   From their TL;DR: 

    We introduce the convolutional VQGAN to combine both the efficiency of convolutional approaches with the expressive power of transformers, and to combine adversarial with likelihood training in a perceptually meaningful way. The VQGAN learns a codebook of context-rich visual parts, whose composition is then modeled with an autoregressive transformer.
   
   Website: https://compvis.github.io/taming-transformers/

   Arxiv paper: https://arxiv.org/abs/2012.09841

.. [vae_architecture]
   --- **Auto-Encoding Variational Bayes**

   This paper describes the architecture used in the Autoencoder of the stable diffusion pipeline.

   From their abstract:

    First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. 
    Second, we show that for i.i.d. datasets with continuous latent variables per datapoint,
    posterior inference can be made especially efficient by fitting an approximate inference 
    model (also called a recognition model) to the intractable posterior using the proposed 
    lower bound estimator. Theoretical advantages are reflected in experimental results.

   Arxiv pdf: https://arxiv.org/pdf/1312.6114.pdf

.. [score_based_generation]
   --- **Score-based Generative Modeling in Latent Space**

   A paper from Nvidia exploring how to learn a diffusion process in latent space while training the autoencoder and the denoising network simultaneously.

   From their contributions section:

    LSGMs can be trained end-to-end by maximizing the variational lower bound on the data likelihood. Compared to regular score matching, our approach comes with additional challenges, since both the score-based denoising model and its target distribution, formed by the latent space encodings, are learnt simultaneously. To this end, we make the following technical contributions: (i) We derive a new denoising score matching objective that allows us to efficiently learn the VAE model and the latent SGM prior at the same time. (ii) We introduce a new parameterization of the latent space score function, which mixes a Normal distribution with a learnable SGM, allowing the SGM to model only the mismatch between the distribution of latent variables and the Normal prior. (iii) We propose techniques for variance reduction of the training objective by designing a new SDE and by analytically deriving importance sampling schemes, allowing us to stably train deep LSGMs.

   Webpage: https://nvlabs.github.io/LSGM/

.. [PNDM]
   ---**Pseudo numerical methods for diffusion models on manifolds**

   This paper explores using higher order pseudo numerical methods for integrating the SDEs describing the reverse diffusion process. 

   From their abstract:

    Specifically, we figure out how to solve differential equations
    on manifolds and show that DDIMs are simple cases of pseudo numerical methods. We change several classical numerical methods to corresponding pseudo
    numerical methods and find that the pseudo linear multi-step method is the best
    in most situations.

   Paper: arxiv.org/pdf/2202.09778.pdf

.. [langevin_diffusion]
   ---**Score-Based Generative Modeling with Critically-Damped Langevin Diffusion**

   This papers integrates the reverse diffusion process by sampling the velocity of the noise instead of its value following equiationg from Langevin dynamics. This allows for a diffusion process with smoother noise variations.

   From their "technocal contributions" section:

    * We propose CLD, a novel diffusion process for SGMs.
    * We derive a score matching objective for CLD which requires only the score of the conditional distribution of velocity given data.
    * We propose hybrid denoising score matching, a new type of denoising score matching ideally suited for scalable training of CLD-based SGMs.
    * We derive a tailored SDE integrator that enables efficient sampling from CLD-based models.
    * Overall, we provide novel insights into SGMs and point out important new connections to statistical mechanics.

   Webpage: https://nv-tlabs.github.io/CLD-SGM/

.. [brain_generation_paper]
   ---**Brain Imaging Generation with Latent Diffusion Models**

   An example of latent diffusion for biomedical image generation. They don't provide any code.

   From their abstract:

    In this study, we explore using Latent Diffusion Models to generate synthetic images from high-resolution 3D brain images. We used T1w MRI images from the UK Biobank dataset (N=31,740) to train our models to learn about the probabilistic distribution of brain images, conditioned on covariables, such as age, sex, and brain structure volumes. We found that our models created realistic data, and we could use the conditioning variables to control the data generation effectively. Besides that, we created a synthetic dataset with 100,000 brain images and made it openly available to the scientific community.

   From their methods section:

    The compression model was an essential step to allow us to scale to high-resolution medical images.
    We trained the autoencoder with a combination of L1 loss, perceptual loss [34], 
    a patch-based adversarial objective [10], and a KL regularization of the latent 
    space. The encoder maps the brain image to a latent representation with a size of 20 × 28 × 20. 

   Paper: https://arxiv.org/abs/2209.07162

   Dataset: https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b
```
