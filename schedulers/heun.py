schedulers_params = {
    "heundiscretescheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "heundiscretescheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "heundiscretescheduler_beta_schedule": {
        "label": "Beta Scheduler",
        "options": {
            "scaled_linear": "Scaled Linear",
            "linear": "Linear",
        },
        "default": "scaled_linear",
        "group": "heundiscretescheduler",
    },
    "heundiscretescheduler_clip_sample": {
        "label": "Clip Sample",
        "type": "boolean",
        "default": False,
        "group": "heundiscretescheduler",
    },
    "heundiscretescheduler_clip_sample_range": {
        "label": "Clip Sample",
        "type": "float",
        "default": 1.0,
        "min": 1.0,
        "max": 10,
        "group": "heundiscretescheduler",
    },
    "heundiscretescheduler_sigmas": {
        "label": "Sigmas",
        "display": "radio",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "default": "default",
        "group": "heundiscretescheduler",
    },
    "heundiscretescheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "default": "leading",
        "group": "heundiscretescheduler",
    },
    "heundiscretescheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "heundiscretescheduler",
    },
    # TODO: check rescale betas SNR
}

scheduler_entry = ("HeunDiscreteScheduler", "heun")
