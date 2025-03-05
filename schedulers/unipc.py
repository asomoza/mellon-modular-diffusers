schedulers_params = {
    "unipcscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "unipcscheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "unipcscheduler_lower_order_final": {
        "label": "Lower Order Final",
        "type": "boolean",
        "default": True,
        "group": "unipcscheduler",
    },
    "dpmsolvermultistepscheduler_sigmas": {
        "label": "Sigmas",
        "display": "radio",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "default": "default",
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_final_sigmas_type": {
        "label": "Final Sigmas Type",
        "options": {
            "zero": "Zero",
            "sigma_min": "Last Sigma",
        },
        "default": "zero",
        "group": "dpmsolvermultistepscheduler",
    },
    "unipcscheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "default": "leading",
        "group": "unipcscheduler",
    },
    "unipcscheduler_solver_order": {
        "label": "Solver Order",
        "type": "int",
        "default": 3,
        "min": 2,
        "max": 3,
        "group": "unipcscheduler",
    },
    "unipcscheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "unipcscheduler",
    },
}

scheduler_entry = ("UniPCMultistepScheduler", "unipc")
