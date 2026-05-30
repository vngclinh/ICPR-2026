"""Multi-architecture pipeline for the LPR ensemble.

Four heterogeneous OCR models (SVTR, new SVTR with SR head, ResTran, CRNN)
share the same training recipe (bf16 AMP, CTC + attention-decoder + SR-MSE
loss schedule, AdamW + OneCycleLR). The final submission is produced by
``eval_multi_arch.py`` which logit-averages and majority-votes over all four.
"""
