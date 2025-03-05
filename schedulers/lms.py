schedulers_params = {
    "lmsdiscretescheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "lmsdiscretescheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "lmsdiscretescheduler_sigmas": {
        "label": "Sigmas",
        "display": "radio",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "default": "default",
        "group": "lmsdiscretescheduler",
    },
    "lmsdiscretescheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "default": "leading",
        "group": "lmsdiscretescheduler",
    },
    "lmsdiscretescheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "lmsdiscretescheduler",
    },
    # TODO: check rescale betas SNR
}

scheduler_entry = ("LMSDiscreteScheduler", "lms")
