schedulers_params = {
    "deismultistepscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "deismultistepscheduler",
            "label": "Advanced",
            "display": "collapse",
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
    "deismultistepscheduler_sigmas": {
        "label": "Sigmas",
        "display": "radio",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "default": "default",
        "group": "deismultistepscheduler",
    },
    # TODO: check rescale betas SNR
}

scheduler_entry = ("DEISMultistepScheduler", "deis")
