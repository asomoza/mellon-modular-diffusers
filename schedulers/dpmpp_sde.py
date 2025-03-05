schedulers_params = {
    "dpmsolversdescheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "dpmsolversdescheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "dpmsolversdescheduler_sigmas": {
        "label": "Sigmas",
        "display": "radio",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "default": "default",
        "group": "dpmsolversdescheduler",
    },
    "dpmsolversdescheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "default": "leading",
        "group": "dpmsolversdescheduler",
    },
    "dpmsolversdescheduler_noise_sampler_seed": {
        "label": "Noise Sampler Seed",
        "type": "int",
        "default": 0,
        "group": "dpmsolversdescheduler",
        "hidden": True,
    },
    "dpmsolversdescheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "dpmsolversdescheduler",
    },
    # TODO: check rescale betas SNR
}

scheduler_entry = ("DPMSolverSDEScheduler", "dpmpp_sde")
