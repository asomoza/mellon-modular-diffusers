from utils.hf_utils import list_local_models
from utils.torch_utils import default_device, device_list, str_to_dtype


class Scheduler:
    def __init__(self, name, scheduler_class, scheduler_args):
        self.name = name
        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_args


schedulers = {
    "DDIMScheduler": "DDIM",
    "DDPMScheduler": "DDPM",
    "DEISMultistepScheduler": "DEIS",
    "DPMSolverSinglestepScheduler": "DPM++ 2S",
    "DPMSolverMultistepScheduler": "DPM++ 2M",
    "DPMSolverSDEScheduler": "DPM++ SDE",
    "EDMDPMSolverMultistepScheduler": "DPM++ 2M EDM",
    "EulerDiscreteScheduler": "Euler",
    "EulerAncestralDiscreteScheduler": "Euler Ancestral",
    "HeunDiscreteScheduler": "Heun",
    "KDPM2DiscreteScheduler": "KDPM2",
    "KDPM2AncestralDiscreteScheduler": "KDPM2 Ancestral",
    "LCMScheduler": "LCM",
    "LMSDiscreteScheduler": "LMS",
    "PNDMScheduler": "PNDM",
    "TCDScheduler": "TCD",
    "UniPCMultistepScheduler": "UniPC",
}


MODULE_MAP = {
    "ControlnetModelLoader": {
        "label": "Load Controlnet Model",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
                "type": "string",
                "default": "xinsir/controlnet-depth-sdxl-1.0",
            },
            "variant": {
                "label": "Variant",
                "type": "string",
                "options": ["", "fp16"],
                "default": "",
                "display": "autocomplete",
                "no_validation": True,
            },
            "dtype": {
                "label": "dtype",
                "options": ["auto", "float32", "bfloat16", "float16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "controlnet_model": {
                "label": "Controlnet Model",
                "display": "output",
                "type": "controlnet_model",
            },
        },
    },
    "ControlnetUnionModelLoader": {
        "label": "Load Controlnet Union Model",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
                "type": "string",
                "default": "OzzyGT/controlnet-union-promax-sdxl-1.0",
            },
            "variant": {
                "label": "Variant",
                "type": "string",
                "options": ["", "fp16"],
                "default": "",
                "display": "autocomplete",
                "no_validation": True,
            },
            "dtype": {
                "label": "dtype",
                "options": ["auto", "float32", "bfloat16", "float16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "controlnet_union_model": {
                "label": "Controlnet Union Model",
                "display": "output",
                "type": "controlnet_union_model",
            },
        },
    },
    "UnetLoader": {
        "label": "Unet Loader",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
                "options": list_local_models(),
                "display": "autocomplete",
                "no_validation": True,
                "default": "stabilityai/stable-diffusion-xl-base-1.0",
            },
            "subfolder": {
                "label": "Subfolder",
                "type": "string",
                "default": "unet",
            },
            "variant": {
                "label": "Variant",
                "type": "string",
                "options": ["", "fp16"],
                "default": "",
                "display": "autocomplete",
                "no_validation": True,
            },
            "dtype": {
                "label": "dtype",
                "options": ["auto", "float32", "bfloat16", "float16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "unet": {
                "label": "Unet",
                "display": "output",
                "type": "diffusers_unet",
            },
        },
    },
    "VAELoader": {
        "label": "VAE Loader",
        "category": "Modular Diffusers",
        "params": {
            "model_id": {
                "label": "Model ID",
                "options": list_local_models(),
                "display": "autocomplete",
                "no_validation": True,
                "default": "stabilityai/stable-diffusion-xl-base-1.0",
            },
            "subfolder": {
                "label": "Subfolder",
                "type": "string",
                "default": "vae",
            },
            "variant": {
                "label": "Variant",
                "type": "string",
                "options": ["", "fp16"],
                "default": "",
                "display": "autocomplete",
                "no_validation": True,
            },
            "dtype": {
                "label": "dtype",
                "options": ["auto", "float32", "bfloat16", "float16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "vae": {
                "label": "VAE",
                "display": "output",
                "type": "diffusers_vae",
            },
        },
    },
    "ModelsLoader": {
        "label": "Diffusers Model Loader",
        "category": "Modular Diffusers",
        "params": {
            "repo_id": {
                "label": "Repository ID",
                "options": list_local_models(),
                "display": "autocomplete",
                "no_validation": True,
                "default": "stabilityai/stable-diffusion-xl-base-1.0",
            },
            "variant": {
                "label": "Variant",
                "type": "string",
                "options": ["", "fp16"],
                "default": "",
                "display": "autocomplete",
                "no_validation": True,
            },
            "device": {
                "label": "Device",
                "type": "string",
                "options": device_list,
                "default": default_device,
            },
            "dtype": {
                "label": "dtype",
                "options": ["auto", "float32", "bfloat16", "float16"],
                "default": "float16",
                "postProcess": str_to_dtype,
            },
            "unet": {
                "label": "Unet",
                "display": "input",
                "type": "diffusers_unet",
            },
            "vae": {
                "label": "VAE",
                "display": "input",
                "type": "diffusers_vae",
            },
            "lora_list": {
                "label": "Lora",
                "display": "input",
                "type": "lora",
            },
            "text_encoders": {
                "label": "Text Encoders",
                "display": "output",
                "type": "diffusers_text_encoders",
            },
            "unet_out": {
                "label": "UNet",
                "display": "output",
                "type": "diffusers_unet",
            },
            "vae_out": {
                "label": "VAE",
                "display": "output",
                "type": "diffusers_vae",
            },
            "scheduler": {
                "label": "Scheduler",
                "display": "output",
                "type": "diffusers_scheduler",
            },
        },
    },
    "EncodePrompt": {
        "label": "Encode Prompt",
        "category": "Modular Diffusers",
        "params": {
            "text_encoders": {
                "label": "Text Encoders",
                "display": "input",
                "type": "diffusers_text_encoders",
            },
            "prompt": {
                "label": "Prompt",
                "type": "string",
                "display": "textarea",
            },
            "negative_prompt": {
                "label": "Negative Prompt",
                "type": "string",
                "display": "textarea",
            },
            "embeddings": {
                "label": "Embeddings",
                "display": "output",
                "type": "prompt_embeddings",
            },
        },
    },
    "Scheduler": {
        "label": "Scheduler",
        "category": "Modular Diffusers",
        "params": {
            "input_scheduler": {
                "label": "Scheduler",
                "display": "input",
                "type": "diffusers_scheduler",
            },
            "scheduler": {
                "label": "Scheduler",
                "display": "select",
                "type": "string",
                "options": schedulers,
                "default": "EulerDiscreteScheduler",
            },
            "karras": {
                "label": "Karras",
                "type": "boolean",
                "default": False,
            },
            "trailing": {
                "label": "Trailing",
                "type": "boolean",
                "default": False,
            },
            "v_prediction": {
                "label": "V-Prediction",
                "type": "boolean",
                "default": False,
            },
            "output_scheduler": {
                "label": "Scheduler",
                "display": "output",
                "type": "diffusers_scheduler",
            },
        },
    },
    "Denoise": {
        "label": "Denoise",
        "category": "Modular Diffusers",
        "params": {
            "unet": {
                "label": "Unet",
                "display": "input",
                "type": "diffusers_unet",
            },
            "scheduler": {
                "label": "Scheduler",
                "display": "input",
                "type": "diffusers_scheduler",
            },
            "embeddings": {
                "label": "Embeddings",
                "display": "input",
                "type": "prompt_embeddings",
            },
            "steps": {
                "label": "Steps",
                "type": "int",
                "default": 25,
                "min": 1,
                "max": 1000,
            },
            "cfg": {
                "label": "Guidance",
                "type": "float",
                "display": "slider",
                "default": 7.0,
                "min": 0,
                "max": 20,
            },
            "seed": {
                "label": "Seed",
                "type": "int",
                "default": 0,
                "min": 0,
                "display": "random",
            },
            "width": {
                "label": "Width",
                "type": "int",
                "display": "text",
                "default": 1024,
                "min": 8,
                "max": 8192,
                "step": 8,
                "group": "dimensions",
            },
            "height": {
                "label": "Height",
                "type": "int",
                "display": "text",
                "default": 1024,
                "min": 8,
                "max": 8192,
                "step": 8,
                "group": "dimensions",
            },
            "guider": {
                "label": "Optional Guider",
                "type": "guider",
                "display": "input",
            },
            "controlnet": {
                "label": "Controlnet",
                "type": "controlnet",
                "display": "input",
            },
            "ip_adapter_image_embeddings": {
                "label": "IP Adapter Embeddings",
                "type": "ip_adapter_embeddings",
                "display": "input",
            },
            "latents": {
                "label": "Latents",
                "type": "latents",
                "display": "output",
            },
        },
    },
    "DecodeLatents": {
        "label": "Decode Latents",
        "category": "Modular Diffusers",
        "params": {
            "vae": {
                "label": "VAE",
                "display": "input",
                "type": "diffusers_vae",
            },
            "latents": {
                "label": "Latents",
                "type": "latents",
                "display": "input",
            },
            "images": {
                "label": "Images",
                "type": "image",
                "display": "output",
            },
        },
    },
    "Lora": {
        "label": "Lora",
        "category": "Modular Diffusers",
        "params": {
            "path": {
                "label": "Path",
                "type": "string",
            },
            "scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 1.0,
                "min": -10,
                "max": 10,
                "step": 0.1,
            },
            "lora": {
                "label": "Lora",
                "type": "lora",
                "display": "output",
            },
        },
    },
    "MultiLora": {
        "label": "Multi Lora",
        "category": "Modular Diffusers",
        "params": {
            "lora_list": {
                "label": "Lora",
                "display": "input",
                "type": "lora",
                "spawn": True,
            },
            "lora": {
                "label": "Multi Loras",
                "type": "lora",
                "display": "output",
            },
        },
    },
    "PAGOptionalGuider": {
        "label": "Perturbed Attention Guidance",
        "category": "Modular Diffusers",
        "params": {
            "pag_scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 3,
                "min": 0,
                "max": 5,
            },
            "pag_layers": {
                "label": "PAG Layers",
                "type": "string",
                "default": "mid",
            },
            "guider": {
                "label": "Guider",
                "display": "output",
                "type": "guider",
            },
        },
    },
    "APGOptionalGuider": {
        "label": "Adaptive Projected Guidance",
        "category": "Modular Diffusers",
        "params": {
            "rescale_factor": {
                "label": "Rescale Factor",
                "type": "float",
                "display": "slider",
                "default": 15,
                "min": 0,
                "max": 20,
                "step": 0.1,
            },
            "momentum": {
                "label": "Momentum",
                "type": "float",
                "display": "slider",
                "default": -0.5,
                "min": -5,
                "max": 5,
                "step": 0.1,
            },
            "guider": {
                "label": "Guider",
                "display": "output",
                "type": "guider",
            },
        },
    },
    "Controlnet": {
        "label": "Controlnet",
        "category": "Modular Diffusers",
        "params": {
            "control_image": {
                "label": "Control Image",
                "type": "image",
                "display": "input",
            },
            "controlnet_conditioning_scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 0.5,
                "min": 0,
                "max": 1,
            },
            "control_guidance_start": {
                "label": "Start",
                "type": "float",
                "display": "slider",
                "default": 0.0,
                "min": 0,
                "max": 1,
            },
            "control_guidance_end": {
                "label": "End",
                "type": "float",
                "display": "slider",
                "default": 1.0,
                "min": 0,
                "max": 1,
            },
            "controlnet_model": {
                "label": "Controlnet Model",
                "type": "controlnet_model",
                "display": "input",
            },
            "controlnet": {
                "label": "Controlnet",
                "display": "output",
                "type": "controlnet",
            },
        },
    },
    "ControlnetUnion": {
        "label": "Controlnet Union",
        "category": "Modular Diffusers",
        "params": {
            "pose_image": {
                "label": "Pose image",
                "type": "image",
                "display": "input",
            },
            "depth_image": {
                "label": "Depth image",
                "type": "image",
                "display": "input",
            },
            "edges_image": {
                "label": "Edges image",
                "type": "image",
                "display": "input",
            },
            "lines_image": {
                "label": "Lines image",
                "type": "image",
                "display": "input",
            },
            "normal_image": {
                "label": "Normal image",
                "type": "image",
                "display": "input",
            },
            "segment_image": {
                "label": "Segment image",
                "type": "image",
                "display": "input",
            },
            "tile_image": {
                "label": "Tile image",
                "type": "image",
                "display": "input",
            },
            "repaint_image": {
                "label": "Repaint image",
                "type": "image",
                "display": "input",
            },
            "controlnet_conditioning_scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 0.5,
                "min": 0,
                "max": 1,
            },
            "control_guidance_start": {
                "label": "Start",
                "type": "float",
                "display": "slider",
                "default": 0.0,
                "min": 0,
                "max": 1,
            },
            "control_guidance_end": {
                "label": "End",
                "type": "float",
                "display": "slider",
                "default": 1.0,
                "min": 0,
                "max": 1,
            },
            "controlnet_model": {
                "label": "Controlnet Union Model",
                "type": "controlnet_union_model",
                "display": "input",
            },
            "controlnet": {
                "label": "Controlnet",
                "display": "output",
                "type": "controlnet",
            },
        },
    },
    "MultiControlNet": {
        "label": "Multi ControlNet",
        "category": "Modular Diffusers",
        "params": {
            "controlnet_list": {
                "label": "ControlNet",
                "display": "input",
                "type": "controlnet",
                "spawn": True,
            },
            "controlnet": {
                "label": "Multi Controlnet",
                "type": "controlnet",
                "display": "output",
            },
        },
    },
    "IPAdapterInput": {
        "label": "IP-Adapter Input",
        "description": "Configure IP-Adapter settings and input",
        "category": "Modular Diffusers",
        "params": {
            "repo_id": {
                "label": "Repository ID",
                "type": "string",
                "default": "h94/IP-Adapter",
            },
            "subfolder": {
                "label": "Subfolder",
                "type": "string",
                "default": "sdxl_models",
            },
            "weight_name": {
                "label": "Weight Name",
                "type": "string",
                "default": "ip-adapter_sdxl_vit-h.safetensors",
            },
            "image_encoder_path": {
                "label": "Image Encoder Path",
                "type": "string",
                "default": "models/image_encoder",
            },
            "image": {
                "label": "Image",
                "display": "input",
                "type": "image",
            },
            "scale": {
                "label": "Scale",
                "type": "float",
                "default": 1.0,
                "min": 0,
                "max": 1,
                "step": 0.01,
            },
            "ip_adapter_input": {
                "label": "IP-Adapter Input",
                "display": "output",
                "type": "ip_adapter",
            },
        },
    },
    "IPAdapter": {
        "label": "IP-Adapter",
        "description": "Process images with IP-Adapter",
        "category": "Modular Diffusers",
        "params": {
            "unet": {
                "label": "UNet",
                "display": "input",
                "type": "diffusers_unet",
            },
            "ip_adapter_inputs": {
                "label": "IP-Adapter Inputs",
                "display": "input",
                "type": "ip_adapter",
                "spawn": True,
            },
            "ip_adapter_image_embeddings": {
                "label": "IP-Adapter Embeddings",
                "display": "output",
                "type": "ip_adapter_embeddings",
            },
            "unet_out": {
                "label": "UNet",
                "display": "output",
                "type": "diffusers_unet",
            },
        },
    },
}
