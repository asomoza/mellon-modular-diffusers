schedulers_params = {
    "dpmsolvermultistepscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "dpmsolvermultistepscheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "dpmsolvermultistepscheduler_algorithm_type": {
        "label": "Algorithm Type",
        "options": {
            "dpmsolver++": "dpm++",
            "sde-dpmsolver++": "sde++",
            "dpmsolver": "dpm",
            "sde-dpmsolver": "sde",
        },
        "default": "dpmsolver++",
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_solver_type": {
        "label": "2nd Order Solver Type",
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
    "dpmsolvermultistepscheduler_use_lu_lambdas": {
        "label": "use Lu Lambdas",
        "type": "boolean",
        "default": False,
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
    "dpmsolvermultistepscheduler_timestep_spacing": {
        "label": "Timestep Spacing",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "default": "leading",
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_solver_order": {
        "label": "Solver Order",
        "type": "int",
        "default": 2,
        "min": 2,
        "max": 3,
        "group": "dpmsolvermultistepscheduler",
    },
    "dpmsolvermultistepscheduler_rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "default": False,
        "group": "dpmsolvermultistepscheduler",
    },
}

scheduler_entry = ("DPMSolverMultistepScheduler", "dpmpp_2m")
