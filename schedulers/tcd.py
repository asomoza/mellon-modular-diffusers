schedulers_params = {
    "tcdscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "tcdscheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "tcdscheduler_beta_schedule": {
        "label": "Beta Scheduler",
        "options": {
            "scaled_linear": "Scaled Linear",
            "linear": "Linear",
            "squaredcos_cap_v2": "Glide Cosine",
        },
        "default": "scaled_linear",
        "group": "tcdscheduler",
    },
    "tcdscheduler_clip_sample": {
        "label": "Clip Sample",
        "type": "boolean",
        "default": False,
        "group": "tcdscheduler",
    },
    "tcdscheduler_clip_sample_range": {
        "label": "Clip Sample",
        "type": "float",
        "default": 1.0,
        "min": 0.1,
        "max": 10,
        "group": "tcdscheduler",
    },
    "tcdscheduler_set_alpha_to_one": {
        "label": "Alpha to One",
        "type": "boolean",
        "default": True,
        "group": "tcdscheduler",
    },
    "tcdscheduler_timestep_scaling": {
        "label": "Timestep Scaling",
        "type": "float",
        "default": 10.0,
        "min": 0.0,
        "max": 100,
        "group": "tcdscheduler",
    },
    "tcdscheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "tcdscheduler",
    },
}

scheduler_entry = ("TCDScheduler", "tcd")
