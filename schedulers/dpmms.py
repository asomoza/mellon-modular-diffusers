schedulers_params = {
    "dpmsolvermultistepscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
            "sample": "Sample",
        },
        "default": "epsilon",
        "group": {
            "key": "dpmsolvermultistepscheduler",
            "label": "Advanced",
            "display": "group",
            "direction": "column",
        },
    },
    "dpmsolvermultistepscheduler_algorithm_type": {
        "label": "Prediction Type",
        "options": {
            "dpmsolver++": "DPM++",
            "sde-dpmsolver++": "SDE++",
            "dpmsolver": "DPM",
            "sde-dpmsolver": "SDE",
        },
        "default": "dpmsolver++",
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_solver_type": {
        "label": "Prediction Type",
        "options": {
            "midpoint": "Midpoint",
            "heun": "Heun",
        },
        "default": "midpoint",
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_euler_at_final": {
        "label": "Euler At Final",
        "type": "boolean",
        "default": False,
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_use_karras_sigmas": {
        "label": "use Karras Sigmas",
        "type": "boolean",
        "default": False,
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_use_exponential_sigmas": {
        "label": "use Exponential Sigmas",
        "type": "boolean",
        "default": False,
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_use_beta_sigmas": {
        "label": "use betas Sigmas",
        "type": "boolean",
        "default": False,
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_use_lu_lambdas": {
        "label": "use Lu Lambdas",
        "type": "boolean",
        "default": False,
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_final_sigmas_type": {
        "label": "Prediction Type",
        "options": {
            "zero": "Zero",
            "sigma_min": "Last Sigma",
        },
        "default": "zero",
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "linspace": "Linspace",
            "leading": "Leading",
            "trailing": "Trailing",
        },
        "default": "linspace",
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "dpmsolvermultistepscheduler",
    },
}

scheduler_entry = ("DPMSolverMultistepScheduler", "DPM++ 2M")
