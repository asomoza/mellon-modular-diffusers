schedulers_params = {
    "eulerdiscretescheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "eulerdiscretescheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "eulerdiscretescheduler_sigmas": {
        "label": "Sigmas",
        "display": "radio",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "default": "default",
        "group": "eulerdiscretescheduler",
    },
    "eulerdiscretescheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "default": "leading",
        "group": "eulerdiscretescheduler",
    },
    "eulerdiscretescheduler_final_sigmas_type": {
        "label": "Final Sigmas Type",
        "options": {
            "zero": "Zero",
            "sigma_min": "Last Sigma",
        },
        "default": "zero",
        "group": "eulerdiscretescheduler",
    },
    "eulerdiscretescheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "eulerdiscretescheduler",
    },
}

scheduler_entry = ("EulerDiscreteScheduler", "euler")
