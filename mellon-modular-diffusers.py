import logging
import os
import time

import torch
from diffusers import (
    AutoModel,
    AdaptiveProjectedGuidance, 
    ClassifierFreeGuidance, 
    SkipLayerGuidance, 
    LayerSkipConfig,
    StableDiffusionXLAutoPipeline, 
    ComponentSpec, 
    StableDiffusionXLModularLoader
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from transformers import CLIPVisionModelWithProjection

from mellon.NodeBase import NodeBase, are_different

# Configure logger
logger = logging.getLogger("mellon")
logger.setLevel(logging.DEBUG)



auto_blocks =StableDiffusionXLAutoPipeline()
text_block = auto_blocks.blocks.pop("text_encoder")
decoder_block = auto_blocks.blocks.pop("decoder")
ip_adapter_block = auto_blocks.blocks.pop("ip_adapter")
image_encoder_block = auto_blocks.blocks.pop("image_encoder")

# Initialize components manager
# components = ComponentsManager()
from custom import components


def _has_changed(old_params, new_params):
    for key in new_params:
        new_value = new_params.get(key)
        old_value = old_params.get(key)

        if new_value is not None and key not in old_params:
            return True
        if are_different(old_value, new_value):
            return True
    return False


def check_nested_changes(old_params, new_params, key_path=None):
    """
    Check if values have changed at a specific nested key path.

    Args:
        old_params: Original parameters dictionary
        new_params: New parameters dictionary
        key_path: String or list of strings representing the nested key path to check.
                 If None, checks at root level.

    Returns:
        bool: True if values at the specified path have changed
    """
    if not key_path:
        return _has_changed(old_params, new_params)

    # Convert string path to list
    keys = key_path.split(".") if isinstance(key_path, str) else key_path

    # Get the values at the specified path
    def get_nested_value(params, keys):
        value = params
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    old_value = get_nested_value(old_params, keys)
    new_value = get_nested_value(new_params, keys)

    # Compare the values
    return are_different(old_value, new_value)


def combine_multi_inputs(inputs):
    """
    Recursively combines a list of dictionaries into a dictionary of lists.
    Handles nested dictionaries by recursively combining their values.

    Args:
        inputs: List of dictionaries to combine

    Returns:
        Dictionary where each key maps to either a list of values or a nested dictionary
        of combined values

    Example:
        inputs = [
            {
                "config": {"repo_id": "repo1", "subfolder": "unet"},
                "scale": 0.5
            },
            {
                "config": {"repo_id": "repo2", "subfolder": "vae"},
                "scale": 0.7
            }
        ]
        result = {
            "config": {
                "repo_id": ["repo1", "repo2"],
                "subfolder": ["unet", "vae"]
            },
            "scale": [0.5, 0.7]
        }
    """
    if not inputs:
        return {}

    # Get all unique keys from all dictionaries
    all_keys = set()
    for d in inputs:
        all_keys.update(d.keys())

    # Initialize the result dictionary
    result = {}

    # Process each key
    for key in all_keys:
        # Get all values for this key
        values = [d.get(key) for d in inputs]

        # If all values are dictionaries, recursively combine them
        if all(isinstance(v, dict) for v in values if v is not None):
            nested_values = [v for v in values if v is not None]
            if nested_values:  # Only combine if there are non-None values
                result[key] = combine_multi_inputs(nested_values)
        else:
            # For non-dictionary values, store as a list
            if any(
                v is not None for v in values
            ):  # Only include if at least one non-None value
                result[key] = values

    return result


# YiYi TODO: add it to diffusers


def update_lora_adapters(lora_node, lora_list):
    """
    Update LoRA adapters based on the provided list of LoRAs.

    Args:
        lora_node: ModularPipeline node containing LoRA functionality
        lora_list: List of dictionaries or single dictionary containing LoRA configurations with:
                  {'lora_path': str, 'weight_name': str, 'adapter_name': str, 'scale': float}
    """
    # Convert single lora to list if needed
    if not isinstance(lora_list, list):
        lora_list = [lora_list]

    # Get currently loaded adapters
    loaded_adapters = list(set().union(*lora_node.get_list_adapters().values()))

    # Determine which adapters to set and remove
    to_set = [lora["adapter_name"] for lora in lora_list]
    to_remove = [adapter for adapter in loaded_adapters if adapter not in to_set]

    # Remove unused adapters first
    for adapter_name in to_remove:
        lora_node.delete_adapters(adapter_name)

    # Load new LoRAs and set their scales
    scales = {}
    for lora in lora_list:
        adapter_name = lora["adapter_name"]
        if adapter_name not in loaded_adapters:
            lora_node.load_lora_weights(
                lora["lora_path"],
                weight_name=lora["weight_name"],
                adapter_name=adapter_name,
            )
        scales[adapter_name] = lora["scale"]

    # Set adapter scales
    if scales:
        lora_node.set_adapters(list(scales.keys()), list(scales.values()))

# YiYi TODO: add to node wrapper class
# "unet" -> "unet_12344"
def node_get_component_id(node_id=None, manager=None, name=None):
    comp_ids = manager._lookup_ids(name=name, collection=node_id)
    if len(comp_ids) != 1:
        raise ValueError(f"Expected 1 component for {name} for node {node_id}, got {len(comp_ids)}")
    return list(comp_ids)[0]

# "unet" 
# -> 
# {"model_id": "unet_12344", 
#  "added_time": 1716239234.0, 
#  "collection": "node_id", 
#  "class_name": "UNet2DConditionModel", 
#  "size_gb": 1.23, 
#  "adapters": ["lora_1", "lora_2"],
#  "ip_adapter": {"", [0.5]}
# }
def node_get_component_info(node_id=None, manager=None, name=None):
    comp_id = node_get_component_id(node_id, manager, name)
    return manager.get_model_info(comp_id)



# (1) loader nodes: nodes that creates components (load models or create components e.g. guiders, schedulers, etc.)
# AutoModelLoader
# ModelsLoader
# Scheduler
# IPAdapterLoader (it also runs the image_encoder)

class AutoModelLoader(NodeBase):
    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove(comp_id)
        super().__del__()

    def execute(self, name, model_id, subfolder, variant, dtype):
        # Debug logging
        logger.debug(f"AutoModelLoader ({self.node_id}) received parameters:")
        logger.debug(f"  name: '{name}'")
        logger.debug(f"  model_id: '{model_id}'")
        logger.debug(f"  subfolder: '{subfolder}'")
        logger.debug(f"  variant: '{variant}'")
        logger.debug(f"  dtype: '{dtype}'")
        
        # Normalize parameters
        variant = None if variant == "" else variant
        subfolder = None if subfolder == "" else subfolder

        spec = ComponentSpec(name=name, repo=model_id, subfolder=subfolder, variant=variant)
        model = spec.load(torch_dtype=dtype)
        comp_id = components.add(name, model, collection=self.node_id)
        logger.debug(f" AutoModelLoader: comp_id added: {comp_id}")
        logger.debug(f" AutoModelLoader: component manager: {components}")

        return {
            "model": components.get_model_info(comp_id)
        }

class ModelsLoader(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        self.loader_class = StableDiffusionXLModularLoader
        self.loader = None

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove(comp_id)
        self.loader = None
        super().__del__()

    def __call__(self, **kwargs):
        self._old_params = self.params.copy()
        return super().__call__(**kwargs)

    def execute(
        self,
        repo_id,
        variant,
        device,
        dtype,
        unet=None,
        vae=None,
        lora_list=None,
    ):
        # Normalize parameters
        variant = None if variant == "" else variant

        def _has_changed(old_params, new_params):
            for key in new_params:
                new_value = new_params.get(key)
                old_value = old_params.get(key)
                if new_value is not None and key not in old_params:
                    return True
                if are_different(old_value, new_value):
                    return True
            return False

        logger.debug(f"""
            ModelsLoader ({self.node_id}) received parameters:
            old_params: {self._old_params}
            new params:
            - repo_id: {repo_id}
            - variant: {variant}
            - dtype: {dtype}
            - unet: {unet}
            - vae: {vae}
            - lora_list: {lora_list}
        """)

        repo_changed = _has_changed(
            self._old_params, {"repo_id": repo_id, "variant": variant, "dtype": dtype}
        )
        unet_input_changed = _has_changed(self._old_params, {"unet": unet})
        vae_input_changed = _has_changed(self._old_params, {"vae": vae})
        lora_input_changed = _has_changed(self._old_params, {"lora_list": lora_list})

        logger.debug(
            f"Changes detected - repo: {repo_changed}, unet: {unet_input_changed}, vae: {vae_input_changed}, "
            f"lora: {lora_input_changed}"
        )

        unet_changed = unet_input_changed or (unet is None and repo_changed)
        vae_changed = vae_input_changed or (vae is None and repo_changed)

        if repo_changed:
            self.loader = self.loader_class.from_pretrained(repo_id, component_manager=components, collection=self.node_id)
            logger.debug(f" ModelsLoader: loader created/updated: {self.loader}")


        # Load and update base models
        loaded_components = {}
        if unet is None:
            if repo_changed or unet_input_changed:
                logger.debug(
                    f" ModelsLoader: load unet from repo_id: {repo_id}, subfolder: unet, variant: {variant}, dtype: {dtype}"
                )
                self.loader.load("unet", variant=variant, torch_dtype=dtype)
        else:
            # unet is always a model info dict if not None
            unet_id = unet["model_id"]
            self.loader.update(unet=components.get_one(unet_id))

        if vae is None:
            if repo_changed or vae_input_changed:
                logger.debug(
                    f" ModelsLoader: load vae from repo_id: {repo_id}, subfolder: vae, variant: {variant}, dtype: {dtype}"
                )
                self.loader.load("vae", variant=variant, torch_dtype=dtype)
        else:
            # vae is always a model info dict if not None
            vae_id = vae["model_id"]
            self.loader.update(vae=components.get_one(vae_id))

        # Load text encoders and scheduler
        if repo_changed:
            logger.debug(
                f" ModelsLoader: load text encoders and scheduler from: {repo_id}, variant: {variant}, dtype: {dtype}"
            )
            self.loader.load(["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"], variant=variant, torch_dtype=dtype)


        # Handle LoRA
        if not lora_list:
            logger.debug(f" ModelsLoader: unload lora:")
            self.loader.unload_lora_weights()
        elif lora_input_changed or unet_changed:
            logger.debug(f" ModelsLoader: load lora:")

            # Unload first to clean previous model's state
            self.loader.unload_lora_weights()
            update_lora_adapters(self.loader, lora_list)

        if unet_changed or vae_changed or repo_changed:
            components.enable_auto_cpu_offload(device=device)

        # Construct loaded_components at the end after all modifications
        loaded_components = {
            "unet_out": node_get_component_info(node_id=self.node_id, manager=components, name="unet"),
            "vae_out": node_get_component_info(node_id=self.node_id, manager=components, name="vae"),
            "text_encoders": {
                k: node_get_component_info(node_id=self.node_id, manager=components, name=k)
                for k in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
            },
            "scheduler": node_get_component_info(node_id=self.node_id, manager=components, name="scheduler"),
        }

        logger.debug(f" ModelsLoader: Final component_manager state: {components}")
        return loaded_components

class Scheduler(NodeBase):
    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove(comp_id)
        super().__del__()

    def execute(self, input_scheduler, scheduler, **kwargs):
        logger.debug(f" Scheduler ({self.node_id}) received parameters:")
        logger.debug(f" - input_scheduler: {input_scheduler}")
        logger.debug(f" - scheduler: {scheduler}")
        logger.debug(f" - kwargs: {kwargs}")

        scheduler_component = components.get_one(input_scheduler["model_id"])

        scheduler_cls = getattr(
            __import__("diffusers", fromlist=[scheduler]), scheduler
        )

        scheduler_options = {}

        # TODO: maybe add some validation, currently assuming that all
        # kwargs are scheduler options
        scheduler_prefix = scheduler.lower()
        for key, value in kwargs.items():
            if "_" in key:
                key_parts = key.split("_", 1)
                if key_parts[0] == scheduler_prefix:
                    if key_parts[1] == "sigmas":
                        if value != "default":
                            scheduler_options[value] = True
                    else:
                        scheduler_options[key_parts[1]] = value

        schedule_spec = ComponentSpec(
            name="scheduler", 
            type_hint=scheduler_cls,
            config=scheduler_component.config, default_creation_method="from_config")
        new_scheduler = schedule_spec.create(**scheduler_options)
        comp_id = components.add(name="scheduler", component=new_scheduler, collection=self.node_id)
        logger.debug(f" Scheduler: new_scheduler: {new_scheduler}")

        return {
            "output_scheduler": components.get_model_info(comp_id)
        }

class IPAdapterLoader(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._ip_adapter_node = ip_adapter_block
        self._ip_adapter_node.setup_loader(component_manager=components, collection=self.node_id)
        self._last_embeddings = None  # Store last valid embeddings

    def __del__(self):
        self._ip_adapter_node.loader.unload_ip_adapter()
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove(comp_id)
        super().__del__()

    def __call__(self, **kwargs):
        # Convert ip_adapter_inputs list to combined map before tracking params
        if "ip_adapter_inputs" in kwargs:
            if not isinstance(kwargs["ip_adapter_inputs"], list):
                kwargs["ip_adapter_inputs"] = [kwargs["ip_adapter_inputs"]]
            kwargs["ip_adapter_inputs"] = combine_multi_inputs(
                kwargs["ip_adapter_inputs"]
            )
        self._old_params = self.params.copy()
        return super().__call__(**kwargs)

    def execute(self, unet, guider, ip_adapter_inputs):
        logger.debug(f" IPAdapterLoader ({self.node_id}) received parameters:")
        logger.debug(f" - unet: {unet}")
        logger.debug(f" - guider: {guider}")
        logger.debug(f" - ip_adapter_inputs: {ip_adapter_inputs}")

        new_params = {"unet": unet, "guider": guider, "ip_adapter_inputs": ip_adapter_inputs}
        unet_changed = check_nested_changes(self._old_params, new_params, "unet")

        ip_adapter_image_changed = check_nested_changes(
            self._old_params, new_params, "ip_adapter_inputs.ip_adapter_image"
        )
        ip_adapter_config_changed = check_nested_changes(
            self._old_params, new_params, "ip_adapter_inputs.ip_adapter_config"
        )
        ip_adapter_scale_changed = check_nested_changes(
            self._old_params, new_params, "ip_adapter_inputs.scale"
        )

        if not ip_adapter_inputs:
            logger.debug(f" unload ip_adapter from components: ip_adapter_{self.node_id}")
            self._ip_adapter_node.unload_ip_adapter()
            self._last_embeddings = None
            return {"ip_adapter_image_embeddings": None, "unet_out": unet}

        repo_ids = ip_adapter_inputs["ip_adapter_config"]["repo_id"]
        subfolders = ip_adapter_inputs["ip_adapter_config"]["subfolder"]
        weight_names = ip_adapter_inputs["ip_adapter_config"]["weight_name"]
        image_encoder_paths = ip_adapter_inputs["ip_adapter_config"][
            "image_encoder_path"
        ]
        scale = ip_adapter_inputs["scale"]
        image = ip_adapter_inputs["ip_adapter_image"]
        
        need_reload = True if ip_adapter_config_changed or unet_changed else False
        if ip_adapter_config_changed:
            # Load the first image encoder (they should all be the same)
            logger.debug(
                f" load image_encoder from repo_id: {repo_ids[0]}, subfolder: {subfolders[0]}, image_encoder_path: {image_encoder_paths[0]}"
            )
            image_encoder_spec = ComponentSpec(name="image_encoder", type_hint=CLIPVisionModelWithProjection, repo=repo_ids[0], subfolder=image_encoder_paths[0])
            image_encoder = image_encoder_spec.load(torch_dtype=torch.float16)
            self._ip_adapter_node.loader.update(image_encoder=image_encoder)

        if unet_changed:
            logger.debug(" update unet")
            self._ip_adapter_node.loader.update(
                unet=components.get_one(unet["model_id"]),
            )
        if guider is not None:
            self._ip_adapter_node.loader.update(guider=guider)
        if need_reload:
            logger.debug(" load ip_adapter(s)")
            self._ip_adapter_node.loader.unload_ip_adapter()
            self._ip_adapter_node.loader.load_ip_adapter(
                repo_ids,
                subfolder=subfolders,
                weight_name=weight_names,
            )

        if ip_adapter_scale_changed:
            logger.debug(f" set ip_adapter scale: {scale}")
            self._ip_adapter_node.loader.set_ip_adapter_scale(scale)

        if ip_adapter_image_changed or need_reload:
            logger.debug(" process ip_adapter image")
            ip_adapter_state = self._ip_adapter_node.run(ip_adapter_image=image)
            self._last_embeddings = ip_adapter_state.intermediates
        
        logger.debug(f" IPAdapterLoader: final component_manager: {components}")

        return {
            "ip_adapter_image_embeddings": self._last_embeddings,
            "unet_out": components.get_model_info(unet["model_id"]),
            "guider_out": guider,
        }

# (2) pipeline nodes: nodes that run (part of) the pipeline (model inference is usually involved)
# EncodePrompt
# Denoise
# DecodeLatents
# IPAdapterInput (it loads also runs the image_encoder)
class EncodePrompt(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._text_encoder_node = text_block
        self._text_encoder_node.setup_loader(component_manager=components)

    def execute(self, guider, text_encoders, **kwargs):
        logger.debug(f" EncodePrompt ({self.node_id}) received parameters:")
        logger.debug(f" - guider: {guider}")
        logger.debug(f" - text_encoders: {text_encoders}")
        logger.debug(f" - kwargs: {kwargs.keys()}")

        text_encoder_components = {
            "text_encoder": components.get_one(text_encoders["text_encoder"]["model_id"]),
            "text_encoder_2": components.get_one(
                text_encoders["text_encoder_2"]["model_id"]
            ),
            "tokenizer": components.get_one(text_encoders["tokenizer"]["model_id"]),
            "tokenizer_2": components.get_one(text_encoders["tokenizer_2"]["model_id"]),
        }

        self._text_encoder_node.loader.update(**text_encoder_components)
        if guider is not None:
            self._text_encoder_node.loader.update(guider=guider)

        logger.debug(f" EncodePrompt: text_encoder_node loader: {self._text_encoder_node.loader}")
        logger.debug(f" EncodePrompt: running text_encoder_node with kwargs: {kwargs}")
        text_state = self._text_encoder_node.run(**kwargs)
        # Return the intermediates dict instead of the PipelineState
        return {"embeddings": text_state.intermediates, "guider_out": guider}


class Denoise(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._denoise_node = auto_blocks
        self._denoise_node.setup_loader(component_manager=components)

    def execute(
        self,
        unet,
        scheduler,
        embeddings,
        steps,
        seed,
        width,
        height,
        guider,
        controlnet,
        ip_adapter_image_embeddings=None,
    ):
        logger.debug(f" Denoise ({self.node_id}) received parameters:")
        logger.debug(f" - unet: {unet}")
        logger.debug(f" - scheduler: {scheduler}")
        logger.debug(f" - embeddings: {embeddings}")
        logger.debug(f" - steps: {steps}")
        logger.debug(f" - seed: {seed}")
        logger.debug(f" - width: {width}")
        logger.debug(f" - height: {height}")
        logger.debug(f" - guider: {guider}")
        logger.debug(f" - controlnet: {controlnet}")
        logger.debug(f" - ip_adapter_image_embeddings: {ip_adapter_image_embeddings}")

        unet_component = components.get_one(unet["model_id"])
        scheduler_component = components.get_one(scheduler["model_id"])
        self._denoise_node.loader.update(
            unet=unet_component,
            scheduler=scheduler_component
        )
        
        if guider is not None:
            self._denoise_node.loader.update(guider=guider)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        denoise_kwargs = {
            **embeddings,  # Now embeddings is already a dict
            "generator": generator,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "output": "latents",
        }

        if ip_adapter_image_embeddings is not None:
            denoise_kwargs.update(
                **ip_adapter_image_embeddings
            )  # Now ip_adapter_image_embeddings is already a dict

        if controlnet is not None:
            denoise_kwargs.update(**controlnet["controlnet_inputs"])

            # For multiple controlnets, get all models from their IDs
            model_ids = controlnet["controlnet_model"]["model_id"]
            if isinstance(model_ids, list):
                controlnet_components = [
                    components.get_one(model_id) for model_id in model_ids
                ]
                controlnet_components = MultiControlNetModel(controlnet_components)
            else:
                controlnet_components = components.get_one(model_ids)

            self._denoise_node.loader.update(controlnet=controlnet_components)

        logger.debug(f" running denoise_node with these kwargs: {denoise_kwargs}")
        latents = self._denoise_node.run(**denoise_kwargs)
        return {"latents": latents}


class DecodeLatents(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._decoder_node = decoder_block
        self._decoder_node.setup_loader(component_manager=components)

    def execute(self, vae, latents):
        logger.debug(f" DecodeLatents ({self.node_id}) received parameters:")
        logger.debug(f" - vae: {vae}")
        logger.debug(f" - latents: {latents.shape}")

        vae_component = components.get_one(vae["model_id"])
        self._decoder_node.loader.update(vae=vae_component)
        images_output = self._decoder_node.run(latents=latents, output="images")
        return {"images": images_output}



# (3) input nodes: nodes that take user inputs and prepare them for other nodes
# Lora
# MultiLora
# PAGOptionalGuider
# Controlnet
# ControlnetUnion
# MultiControlNet
class Lora(NodeBase):
    def execute(self, path, scale, is_local=False):
        if is_local:
            lora_path = os.path.dirname(path)
            weight_name = os.path.basename(path)
        else:
            # Handle hub path format: "org/model_id/filename"
            parts = path.split("/")
            if len(parts) != 3:
                raise ValueError("Hub path must be in format 'org/model_id/filename'")
            lora_path = f"{parts[0]}/{parts[1]}"
            weight_name = parts[2]

        adapter_name = os.path.splitext(weight_name)[0]

        # Return the lora configuration directly, not wrapped in another dict
        return {
            "lora": {
                "lora_path": lora_path,
                "weight_name": weight_name,
                "adapter_name": adapter_name,
                "scale": scale,
            }
        }


class MultiLora(NodeBase):
    def execute(self, lora_list):
        return {"lora": lora_list}


class PAGOptionalGuider(NodeBase):
    def execute(self, guidance_scale, skip_layer_guidance_scale):
        # TODO: Maybe do some validations to ensure correct layers
        peg_config = {
            "guidance_scale": guidance_scale,
            "skip_layer_guidance_scale": skip_layer_guidance_scale,
            "skip_layer_config": LayerSkipConfig(
                indices=[2, 3, 7, 8],
                fqn="mid_block.attentions.0.transformer_blocks",
                skip_attention=False,
                skip_ff=False,
                skip_attention_scores=True,
            ),
            "start": 0.0,
            "stop": 1.0,
        }
        pag_spec = ComponentSpec(name="guider", type_hint=SkipLayerGuidance, config=peg_config, default_creation_method="from_config")
        return {"guider": pag_spec}


class Controlnet(NodeBase):
    def execute(
        self,
        control_image,
        controlnet_conditioning_scale,
        controlnet_model,
        control_guidance_start,
        control_guidance_end,
    ):
        controlnet = {
            "controlnet_model": controlnet_model,
            "controlnet_inputs": {
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            },
        }
        return {"controlnet": controlnet}


class ControlnetUnion(NodeBase):
    def execute(
        self,
        pose_image,
        depth_image,
        edges_image,
        lines_image,
        normal_image,
        segment_image,
        tile_image,
        repaint_image,
        controlnet_conditioning_scale,
        controlnet_model,
        control_guidance_start,
        control_guidance_end,
    ):
        # Map identifiers to their corresponding index and image
        image_map = {
            "pose_image": (pose_image, 0),
            "depth_image": (depth_image, 1),
            "edges_image": (edges_image, 2),
            "lines_image": (lines_image, 3),
            "normal_image": (normal_image, 4),
            "segment_image": (segment_image, 5),
            "tile_image": (tile_image, 6),
            "repaint_image": (repaint_image, 7),
        }

        # Initialize control_mode and control_image
        control_mode = []
        control_image = []

        # Iterate through the dictionary and add non-None images to the lists
        for key, (image, index) in image_map.items():
            if image is not None:
                control_mode.append(index)
                control_image.append(image)

        controlnet = {
            "controlnet_model": controlnet_model,
            "controlnet_inputs": {
                "control_image": control_image,
                "control_mode": control_mode,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            },
        }
        return {"controlnet": controlnet}


class MultiControlNet(NodeBase):
    def execute(self, controlnet_list):
        controlnet = combine_multi_inputs(controlnet_list)
        return {"controlnet": controlnet}


class IPAdapterInput(NodeBase):
    def execute(
        self,
        repo_id,
        subfolder,
        weight_name,
        image_encoder_path,
        image,
        scale,
    ):
        ip_adapter_config = {
            "repo_id": repo_id,
            "subfolder": subfolder,
            "weight_name": weight_name,
            "image_encoder_path": image_encoder_path,
        }
        ip_adapter_input = {
            "ip_adapter_image": image,
            "ip_adapter_config": ip_adapter_config,
            "scale": scale,
        }

        return {"ip_adapter_input": ip_adapter_input}

