schedulers_params = {
    "lcmscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "lcmscheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "lcmscheduler_beta_schedule": {
        "label": "Beta Scheduler",
        "options": {
            "scaled_linear": "Scaled Linear",
            "linear": "Linear",
            "squaredcos_cap_v2": "Glide Cosine",
        },
        "default": "scaled_linear",
        "group": "lcmscheduler",
    },
    "lcmscheduler_clip_sample": {
        "label": "Clip Sample",
        "type": "boolean",
        "default": False,
        "group": "lcmscheduler",
    },
    "lcmscheduler_clip_sample_range": {
        "label": "Clip Sample",
        "type": "float",
        "default": 1.0,
        "min": 1.0,
        "max": 10,
        "group": "lcmscheduler",
    },
    "lcmscheduler_set_alpha_to_one": {
        "label": "Alpha to One",
        "type": "boolean",
        "default": True,
        "group": "lcmscheduler",
    },
    "lcmscheduler_timestep_scaling": {
        "label": "Timestep Scaling",
        "type": "float",
        "default": 10.0,
        "min": 0.0,
        "max": 100,
        "group": "lcmscheduler",
    },
    "lcmscheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "lcmscheduler",
    },
}

scheduler_entry = ("LCMScheduler", "lcm")
