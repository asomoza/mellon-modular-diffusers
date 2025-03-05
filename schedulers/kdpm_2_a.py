schedulers_params = {
    "kdpm2ancestraldiscretescheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "kdpm2ancestraldiscretescheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "kdpm2ancestraldiscretescheduler_sigmas": {
        "label": "Sigmas",
        "display": "radio",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "default": "default",
        "group": "kdpm2ancestraldiscretescheduler",
    },
    "kdpm2ancestraldiscretescheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "default": "leading",
        "group": "kdpm2ancestraldiscretescheduler",
    },
    # TODO: check rescale betas SNR
}

scheduler_entry = ("KDPM2AncestralDiscreteScheduler", "kdpm_2_a")
