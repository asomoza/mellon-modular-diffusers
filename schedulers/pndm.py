schedulers_params = {
    "pndmscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "pndmscheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "pndmscheduler_skip_prk_steps": {
        "label": "Skip Runge-Kutta steps",
        "type": "boolean",
        "default": False,
        "group": "pndmscheduler",
    },
    "pndmscheduler_set_alpha_to_one": {
        "label": "Alpha to One",
        "type": "boolean",
        "default": False,
        "group": "pndmscheduler",
    },
    "pndmscheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
        },
        "default": "leading",
        "group": "pndmscheduler",
    },
    # TODO: check rescale betas SNR
}

scheduler_entry = ("PNDMScheduler", "pndm")
