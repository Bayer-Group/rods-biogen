# Few-shot learning

## Introduction
{cite}`FslSurvey2020` is a nice introduction to FSL. It summarises the core
issue of FSL and presents a useful taxonomy to categorize the different
approaches that have been used to address FSL. It is a fairly comprehensive
review and cites many foundational papers. It is incomplete, though, mainly
because it misses many of the advances that the field has seen since 2020.

## State of the art
The **P>M>F pipeline** {cite}`PmfPipeline2022` suggests that a three-stage
pipeline of pre-training on external data, meta-training with labelled few-shot
tasks, and task-specific fine-tuning on unseen tasks can offer excellent
performance, with the added advantage of being much simpler than many recent
meta-learning approaches.
