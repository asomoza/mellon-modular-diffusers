schedulers_params = {
    "deismultistepscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
            "sample": "Sample",
        },
        "default": "epsilon",
        "group": {
            "key": "deismultistepscheduler",
            "label": "Advanced",
            "display": "group",
            "direction": "column",
        },
    },
    "deismultistepscheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
        },
        "default": "leading",
        "group": "deismultistepscheduler",
    },
    "deismultistepscheduler_use_karras_sigmas": {
        "label": "use Karras Sigmas",
        "type": "boolean",
        "default": False,
        "group": "deismultistepscheduler",
    },
}

scheduler_entry = ("DEISMultistepScheduler", "DEIS")
