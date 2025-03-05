schedulers_params = {
    "kdpm2discretescheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "kdpm2discretescheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "kdpm2discretescheduler_sigmas": {
        "label": "Sigmas",
        "display": "radio",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "default": "default",
        "group": "kdpm2discretescheduler",
    },
    "kdpm2discretescheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "default": "leading",
        "group": "kdpm2discretescheduler",
    },
    # TODO: check rescale betas SNR
}

scheduler_entry = ("KDPM2DiscreteScheduler", "kdpm_2")
