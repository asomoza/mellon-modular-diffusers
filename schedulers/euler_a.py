schedulers_params = {
    "eulerancestraldiscretescheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "eulerancestraldiscretescheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "eulerancestraldiscretescheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "default": "leading",
        "group": "eulerancestraldiscretescheduler",
    },
    "eulerancestraldiscretescheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "eulerancestraldiscretescheduler",
    },
}

scheduler_entry = ("EulerAncestralDiscreteScheduler", "euler_a")
