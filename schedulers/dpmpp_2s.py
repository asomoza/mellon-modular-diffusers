schedulers_params = {
    "dpmsolversinglestepscheduler_prediction_type": {
        "label": "Prediction Type",
        "options": {
            "epsilon": "Epsilon",
            "v_prediction": "V Prediction",
        },
        "default": "epsilon",
        "group": {
            "key": "dpmsolversinglestepscheduler",
            "label": "Advanced",
            "display": "collapse",
        },
    },
    "dpmsolversinglestepscheduler_algorithm_type": {
        "label": "Algorithm Type",
        "options": {
            "dpmsolver++": "dpm++",
            "sde-dpmsolver++": "sde++",
        },
        "default": "dpmsolver++",
        "group": "dpmsolversinglestepscheduler",
    },
    "dpmsolversinglestepscheduler_solver_type": {
        "label": "2nd Order Solver Type",
        "options": {
            "midpoint": "Midpoint",
            "heun": "Heun",
        },
        "default": "midpoint",
        "group": "dpmsolversinglestepscheduler",
    },
    "dpmsolversinglestepscheduler_sigmas": {
        "label": "Sigmas",
        "display": "radio",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "default": "default",
        "group": "dpmsolversinglestepscheduler",
    },
    "dpmsolversinglestepscheduler_final_sigmas_type": {
        "label": "Final Sigmas Type",
        "options": {
            "zero": "Zero",
            "sigma_min": "Last Sigma",
        },
        "default": "zero",
        "group": "dpmsolversinglestepscheduler",
    },
    # TODO: check rescale betas SNR
}

scheduler_entry = ("DPMSolverSinglestepScheduler", "dpmpp_2s")
