from utils.hf_utils import list_local_models
from utils.torch_utils import default_device, device_list, str_to_dtype

from .schedulers import load_all_schedulers

MODULE_MAP = {

    "AutoModelLoader": {
        "label": "Auto Model Loader",
        "category": "Modular Diffusers",
        "params": {
            "name": {
                "label": "Name",
                "type": "string",
            },
            "model_id": {
                "label": "Model ID",
                "type": "string",
            },
            "subfolder": {
                "label": "Subfolder",
                "type": "string",
            },
            "variant": {
                "label": "Variant",
                "type": "string",
                "options": ["", "fp16", "bf16"],
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
            "model": {
                "label": "Model",
                "display": "output",
                "type": "diffusers_auto_model",
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
                "default": "YiYiXu/modular-demo-auto",
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
                "type": "diffusers_auto_model",
            },
            "vae": {
                "label": "VAE",
                "display": "input",
                "type": "diffusers_auto_model",
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
                "type": "diffusers_auto_model",
            },
            "vae_out": {
                "label": "VAE",
                "display": "output",
                "type": "diffusers_auto_model",
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
            "guider": {
                "label": "Guider",
                "display": "input",
                "type": "guider",
            },
            "prompt": {
                "label": "Prompt",
                "type": "string",
                "default": "a bear sitting in a chair drinking a milkshake",
                "display": "textarea",
            },
            "negative_prompt": {
                "label": "Negative Prompt",
                "type": "string",
                "default": "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
                "display": "textarea",
            },
            "embeddings": {
                "label": "Embeddings",
                "display": "output",
                "type": "prompt_embeddings",
            },
                "guider_out": {
                "label": "Guider",
                "display": "output",
                "type": "guider",
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
                "type": "diffusers_auto_model",
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
                "label": "Guider",
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
                "type": "diffusers_auto_model",
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
            "is_local": {
                "label": "Is Local Path",
                "type": "boolean",
                "default": False,
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
            "guidance_scale": {
                "label": "Guidance Scale",
                "type": "float",
                "display": "slider",
                "default": 5.0,
                "min": 0,
                "max": 10,
            },
            "skip_layer_guidance_scale": {
                "label": "Scale",
                "type": "float",
                "display": "slider",
                "default": 3,
                "min": 0,
                "max": 5,
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
                "type": "diffusers_auto_model",
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
                "type": "diffusers_auto_model",
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
    "IPAdapterLoader": {
        "label": "IP-Adapter Loader",
        "description": "IP-Adapter Loader",
        "category": "Modular Diffusers",
        "params": {
            "unet": {
                "label": "UNet",
                "display": "input",
                "type": "diffusers_auto_model",
            },
            "guider": {
                "label": "Guider",
                "display": "input",
                "type": "guider",
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
                "type": "diffusers_auto_model",
            },
            "guider_out": {
                "label": "Guider",
                "display": "output",
                "type": "guider",
            },
        },
    },
}

schedulers, schedulers_params = load_all_schedulers()

scheduler_selection = {
    "scheduler": {
        "label": "Scheduler",
        "options": schedulers,
        "default": list(schedulers.keys())[0],
        "onChange": {
            "action": "show",
            "target": {key: f"{key.lower()}_group" for key in schedulers},
        },
    }
}

MODULE_MAP["Scheduler"]["params"].update(scheduler_selection)
MODULE_MAP["Scheduler"]["params"].update(schedulers_params)
