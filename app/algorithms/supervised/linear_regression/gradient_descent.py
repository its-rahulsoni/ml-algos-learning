import os
import numpy as np
import matplotlib.pyplot as plt

from linear_regression_noise_data import (
    build_design_matrix, generate_noisy_data, fit_closed_form,
    predict, compute_residuals, compute_loss,
)