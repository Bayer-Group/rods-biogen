# Project OKRs

The goal is to provide a pipeline that allows to:

- Train high-performance models on a small data regime.
- Leveraging the synthetic data generation capabilities of diffusion models
- Taking advantage of open sourced implementations to reduce training costs.

## Personal goals

- Create cost-efficient solutions that provide the maximum value possible to stakeholders.
- Establish a solid track record of high impact publications.
- Develop a top-notch codebase that allows us to achieve minimal time-to-marked for our products.
- Build a world-class infrastructure that scales horizontally with minimal friction.

## OKRs for the remaining of 2022 & Q1 2023

In order to achieve our goals, the first step is proposing short term objectives that serve as a steping stone
to create the desired business value. To measure the progress of the project and it alignment with the goals
we set, we propose the following OKRs to track our progress.

### Stakeholder engagement (Project & product management)

#### Objective: Define a portfolio of valuable use-cases

This entails determining a set of business problems that are valuable for stakeholders and possible
to tackle withing the stablished deadlines.

**Key results**:

- Understanding the top 3 valuable areas of interest of our stakeholders, their expectations, and all the necesary business requirements.
- Choosing the one that we can solve faster and design an MVP.
- Build a pitch deck to present our solutions to the stakeholders.

#### Objective: Build engaging demos to improve how stakeholders perceive trangible value in our proposals

**Key results**:

- Build one interactive dashboard for every use case in our pitch deck that showcases our breakthroughs / helps
  stakeholders understand better the value we can provide.
- Integrate the dashboards with our pitch deck to highlight tangible examples of our capabilities. This can be done using images, videos, or live demos of our apps.
- Being capable of adapting our demos to a new domain-specific dataset of images in under 1 hour.
- Deploy a web sever running all the demos of the pitch deck and treat it as production.

#### Objective: Develop a functional MVP that we can offer as a solution

**Key results**:

- Build the simples MVP capable of providing busines value within the scope of a proof of concept (Limited to certain datasets, not considering performance optimizations, etc.)
- Develop a demo application controllable throug a GUI that the stakeholders can try themselves.
- Containerize the app so it can be installed and run with a single command.
- Deploy the app to the demo server and treat it as production.

### Research and Experimentation-Code

#### Objective: Understand the latent space of the stable diffusion VAE

**Key results**:

- Build interactive embedding visualizations using state of the art dimensionality reduction techniques.
- Define and implement a set of metrics for evaluating the quality of image reconstructions.
- Define and implement a set of metrics for masuring latent representations and their relationships.
- Build interactive visualizations for understanding the structure of the latent space and how it relates to different metrics.
- Build interactive visualizations for understanding the reconstructions of latent representations.
- Write a technical report notebook highlighting the relevant findings of each one of the experiments performed.
- Define strategy for improving the technical report notebooks until they reach pitch deck quality.

#### Objective: Understand how interpolations and naive data augmentation techniques work in latent space

**Key results**:

- Experiment with latent augmentation using linear classification inversion.
- Experiment with linear interpolation between latents.
- Experiment with noisy perturbation of latents.
- Experiment with upsampling techniques like SMOTE.
- Write an experiment report notebook on the quality reconstruction of the generated latents.
- Incorporate main highlights into the team pitch deck.

#### Objective: Understand metric-guided generation

**Key results**:

- Experiment with sample-guided generation.
- Experiment with metric guidance using classification loss.
- Train and evaluate the discriminator model.
- Experiment with metric guidance using discrimination loss.
- Experiment with metric guidance using out of distribution loss.

#### Objective: Understand latent representation's potential as a pre-processing step

**Key results**:

- Train at least 5 different sk-learn classifiers in latent space and evaluate their performance.
- Train a transformer on latent space and evaluate its performance.
- Define and implement at least 3 benchmark models widely used on medical image classification.
- Train and evaluate In-Out of distribution classifier.
- Evaluate latent-base classifiers on 2 different datasets.

#### Objective: Achieve a strong publication record

**Key results**:

- Define at least 2 potential publications suitable for at least mid-tier venues or 1 publication suitable for a top5 venue.
- Agree on how to structure our findings we can publish and select an appropriate venue for each of them.
- Upgrade the relevant report nootebooks to publication grade quality.
- Open source a dedicated webpage on GitHub for each publication offering a blog post, code and paper.
- Ensure that the released code meets high quality standards.

### Product developement: First steps towards a common framework to productize the team's code (Coding & workflow)

#### Objective: Minimize iteration time

**Key results**:

- Be able to run one meaningful new experiment every week. It can be either a new approach to an existing method, or the evaluation of new features developed.
- Analyze the results of an experiment in a notebook in less than 20 lines of code.
- Build every new feature writing less than 5% duplicated code.
- Be able to re-run any old experiment with one line of code.
- Create a report on a re-run of an old experiment by running its respective notebook changing only a single cell.
- Onboard the first new developer so they can implement new features in less than a month.

#### Objective: Develop a high quality code base that can be shared across the team

**Key results**:

- Have the main branch of the project always passing all CI/CD checks, and create a new released tag for each commit to main.
- Achieve a test coverage of +90% on all the non-device specific code.
- The main branch passes all the docstring checks and the doc updates deploy automatically.
- The code can be installed and used in any new machine with a single command.
- The experiments are reproducible and dependency management best practices are implemented.

#### Objective: Minimize cognitive load and complexity by building good abstractions

**Key results**:

- Release an alpha version of the team codebase library that abstracts the common research use-cases.
- Include docstrings and at least 1 usage example for all the public facing features.
