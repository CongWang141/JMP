# Counterfactual and Synthetic Control Method: Causal Inference with Instrumented Principal Component Analysis

This repository is dedicated to the paper *"Counterfactual and Synthetic Control Method: Causal Inference with Instrumented Principal Component Analysis."* It contains all the resources necessary to reproduce the empirical study and analyses presented in the paper.

## Repository Structure

### 1. `data/`
The `data` folder contains the datasets used in the empirical study:

- **FDI net inflow to OECD countries**: This data is used in the paper and can also be downloaded from the World Development Indicators (WDI) database.
- **California proposition 99 data**

### 2. `figs/`
The `figs` folder contains all the figures generated in the paper.

### 3. `latex/`
The `latex` folder contains the LaTeX files for the main manuscript of the paper.

### 4. `old_test/`
The `old_test` folder contains earlier tests conducted for this paper. The structure is somewhat disorganized, so it's advisable to ignore this folder.

### 5. `papers/`
The `papers` folder contains some of the related literature and references used in the research.

### 6. `slides/`
The `slides/' folder contains updated slides for discussion.

### 7. `src/`
The `src` folder contains all the scripts and functions required to:

- Generate data
- Conduct the estimation
- Compute confidence intervals

### 8. Test Files
The following test files are well-structured and organized:

- **`test_file`**: Demonstrates different treatment assignment mechanisms.
- **`test1_data_generating`**: Generates the data used in the simulations.
- **`test2_estimation`**: Estimates the treatment effect using different methods.
- **`test3_bias_comparison`**: Compares the bias of different methods.
- **`test4_conformal_inference`**: Computes confidence intervals using conformal inference.
- **`test5_finite_sample`**: Conducts finite sample property analysis.
- **`test6_normalization`**: Demonstrates the process of normalization.
- **`test7_empirical_study`**: Conducts the empirical study on FDI.

## Getting Started

To get started, clone the repository and navigate through the folders as outlined above. Each folder is self-contained with specific resources to assist you in replicating the results presented in the paper.

```bash
git clone https://github.com/CongWang141/JMP.git
cd your-repository-name
