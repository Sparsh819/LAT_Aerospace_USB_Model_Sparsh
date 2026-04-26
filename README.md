# LAT_Aerospace_USB_Model_Sparsh
Engineered a high-fidelity (~90% accurate) USB jet–airfoil model for a thick jet flap on S1223, extending NASA-based theory. Captures jet deflection, Coandă-approximated flow turning, and lift augmentation, achieving strong agreement with CFD and literature across operating regimes.

# USB Jet Modeling for Thick Jet Flap Airfoil 
# Overview

This project presents a high-fidelity mathematical model (~90% accuracy) for simulating Upper Surface Blowing (USB) jet–airfoil interaction in a thick jet flap configuration.

The work extends classical USB aerodynamic theory into a computationally efficient solver, demonstrating strong agreement with CFD simulations and established research results.

# Key Highlights
Approximately 90% agreement with CFD simulations
Based on established USB aerodynamic theory (NASA)
Captures thick jet behavior beyond thin jet assumptions
Uses a camber-line-based vortex panel method
Provides a computationally efficient alternative to CFD

# Problem Context

The objective is to model a powered-lift system capable of achieving very high lift coefficients for STOL applications.

This requires capturing complex aerodynamic phenomena such as:

Jet–airfoil interaction
Flow turning over the flap (Coandă effect)
Lift augmentation due to jet momentum

# Mathematical Model Description

The model is built using a vortex panel method applied to the camber line of the airfoil, coupled with jet interaction physics.

Core Features:
Coupled modeling of jet flow and outer flow interaction
Inclusion of jet momentum effects
Representation of flow deflection over the flap
Numerical solution via panel-based discretization

# Key Assumptions

To ensure tractability while maintaining physical accuracy, the following assumptions are used:

Jet Thickness
The jet thickness is fixed at 86 mm, representing the portion above the airfoil.
Jet Origin
The jet begins at 18.73% chord length, closely matching the physical model value of 18.37%.
Jet Demarcation
The jet is demarcated at 70% chord length, after which it aligns with the flap.
Coandă Effect Approximation
The Coandă effect is approximated using a sharp directional change at the demarcation point to simplify viscous behavior.
Airfoil Selection
The S1223 high-lift airfoil is used throughout the model.
Jet Direction
The jet is assumed to align with the flap after deflection.
Variable Air Density
Air density is treated as an input parameter to allow modeling under different operating conditions.

# Validation

The model has been validated against:

Published theoretical results on USB aerodynamics
CFD simulation outputs
Observations:
Strong agreement in lift coefficient trends
Accurate prediction of jet deflection behavior
Consistent representation of flow–flap interaction

The model performs reliably across both standard conditions and selected edge cases, establishing it as a robust reduced-order aerodynamic tool.

# Code Implementation

The solver is implemented in Python and includes:

Camber-line extraction from airfoil geometry
Panel discretization of airfoil and jet regions
Coupled system solution for circulation and jet interaction
Lift decomposition into:
Circulation-induced lift
Jet reaction-induced lift

# Repository Contents

Research papers and theoretical references
Python implementation of the mathematical model
Validation results comparing model and CFD outputs
Documentation of assumptions and methodology

# Applications

STOL aircraft design and analysis
Powered-lift aerodynamic systems
UAV propulsion–airfoil interaction studies
Preliminary design prior to CFD analysis
Control-oriented aerodynamic modeling
