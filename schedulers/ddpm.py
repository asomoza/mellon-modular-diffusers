schedulers_params = {
    "ddpmscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "ddpmscheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "ddpmscheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
        },
        "default": "leading",
        "group": "ddpmscheduler",
    },
    "ddpmscheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "ddpmscheduler",
    },
}

scheduler_entry = ("DDPMScheduler", "ddpm")
