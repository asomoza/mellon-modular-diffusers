schedulers_params = {
    "ddimscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
            "sample": "Sample",
        },
        "default": "epsilon",
        "group": {
            "key": "ddimscheduler",
            "label": "Advanced",
            "display": "group",
            "direction": "column",
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

scheduler_entry = (
    "DDIMScheduler",
    "DDIM",
)
