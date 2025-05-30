schedulers_params = {
    "ddimscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "ddimscheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "ddimscheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
        },
        "default": "leading",
        "group": "ddimscheduler",
    },
    "ddimscheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "ddimscheduler",
    },
}

scheduler_entry = ("DDIMScheduler", "ddim")
