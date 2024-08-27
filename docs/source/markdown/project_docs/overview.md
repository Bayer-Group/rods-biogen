# Overview

This is an overview of the potential research directions that we can take while figuring out how to
master diffusion models to provide business value, while stablishing a solid track record of relevant publications.

## Value propositions

- **Compliance and privacy**: Avoid many compliance issues that are crucial to stakeholders by training on synthetic datasets.
- **Performance**: Increase model performance & reduce the impact of data scarcity.
- **Explainability**: Allow non-technical users to better understand and interact with our AI solutions.
- **Cost efficiency**: Leveraging pre-trained models massively decreases the costs of developing new solutions.

## Use cases

- Training a DL model purely on synthetic data --> Compliance & privacy, cost efficiency
- Using SD VAE's latent space as a preprocessing step for classification tasks --> Performance & cost efficiency
- Data augmentation for improving model performance when not enough data is available --> Performance & cost efficiency
- GUI for understanding the model's predictions --> Explainability

## Phases of the project

1. **Due diligence**:
   - Understand and reproduce the state of the art.
   - Evaluate the technologies available to develop the project and choose the ones that offer the best tradeoffs of perfomance/simplicity.
   - Define feasible use-cases according to the timeline and stakeholder preferences.

2. **Proof of concept**:
   - Build a demo that showcases the project's value propositions. During this phase we focus on building an appealing "frontend" that helps stakeholders understand the value of the project.
   - Experiment with simple yet potentially valuable approaches without training any DL models.
   - Use the smallest datasets possible.
   - Develop the analysis tools (metrics and visualizations) that we will need in the next phase.

3. **MVP**:
   - Implement the main use cases, stablish a consistent workflow and exponentially increase experimentation capabilities.
   - Train a single DL model at a time on 1 GPU / using DP on 1 node, and avoid experiments that require fine-tuning the stable diffusion modules.
   - Develop the experimentation infrastructure & tools to evaluate and benchmark the models that we train.
   - Test our findings in medium scale datasets, and allow for incorporating new datasets easily.
   - Decide on a publication strategy to publish our most relevant fidings.
   - Deliver a demo web app that stakeholders can easily install and access.

4. **Alpha release**:
   - Scale the use cases of the previous phase to scale efficiently to a multi-GPU single node.(Advanced multi-GPU strategies on 1 node)
   - Experiment with fine-tuning the stable diffusion modules on small / medium scale datasets.
   - Incorporate one large-scale dataset for the experiments that can be run in a single node.
   - Start writing papers and reports about our most relevant fidings.
   - Refactor codebase and stablish a consistent API that allows further scaling.

5. **Beta release**:
   - Scale our experimentations capabilities to multi-node multi-GPU clusters.
   - Adapt our solution to work with datasets from different domains.
   - Run experiments on real-world datasets.
   - Submit most relevant finding to the best journals/conferences.
   - Develop tools to evaluate the real business impact of our solutions.
   - Design release and Open Source strategy to back up our publications and grain traction.

6. **Production**:
   - Allow stakeholders to implement our solutions to their workflows with minimal friction.
   - Decrease the time needed to adapt our solutions to different domains / use cases.
   - Optimize the business value of our solutions.
   - Minimize onboarding time to the project.
   - Share our know how with the rest of the DS team / community.
