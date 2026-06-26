"""Domain-adaptation methods.

Each method exposes a uniform entry point so it can be selected via Hydra:

    adapt(model, source_loader, target_loader, cfg) -> AdaptedModel
"""
