"""Shared library of the RFF backdoor experiments.

- backdoor: key, backdoored ("pancake") rows, RFF model, lattice attack, fine-tuning (Goldwasser et al., Sec. 6).
- encoder_common: public PCA, paper key, planted / honest heads and flip rates for the encoder experiments.

Note: importing backdoor sets torch's default dtype to float64.
"""
