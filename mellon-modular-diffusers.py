import asyncio
import logging
import os

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    ControlNetUnionModel,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    ModularPipeline,
    StableDiffusionXLAutoPipeline,
    UNet2DConditionModel,
)
from diffusers.guider import APGGuider, CFGGuider, PAGGuider
from diffusers.pipelines.components_manager import ComponentsManager
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_modular import (
    AUTO_BLOCKS,
    StableDiffusionXLAutoDecodeStep,
    StableDiffusionXLIPAdapterStep,
    StableDiffusionXLLoraStep,
    StableDiffusionXLTextEncoderStep,
)
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from image_gen_aux import DepthPreprocessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mellon.NodeBase import NodeBase, are_different
from mellon.server import web_server

logger = logging.getLogger("mellon")

# YiYi TODO: make the SDXLAutoBlocks user can directly import
all_blocks_map = AUTO_BLOCKS.copy()
text_block = all_blocks_map.pop("text_encoder")()
decoder_block = all_blocks_map.pop("decode")()
image_encoder_block = all_blocks_map.pop("image_encoder")()


class SDXLAutoBlocks(SequentialPipelineBlocks):
    block_classes = list(all_blocks_map.values())
    block_names = list(all_blocks_map.keys())


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


components = ComponentsManager()


class ControlnetModelLoader(NodeBase):
    def __del__(self):
        components.remove(f"controlnet_{self.node_id}")
        super().__del__()

    def execute(self, model_id, variant, dtype):
        # Normalize parameters
        variant = None if variant == "" else variant

        controlnet_model = ControlNetModel.from_pretrained(
            model_id, variant=variant, torch_dtype=dtype
        )
        components.add(f"controlnet_{self.node_id}", controlnet_model)
        return {
            "controlnet_model": components.get_model_info(f"controlnet_{self.node_id}")
        }


class ControlnetUnionModelLoader(NodeBase):
    def __del__(self):
        components.remove(f"controlnet_{self.node_id}")
        super().__del__()

    def execute(self, model_id, variant, dtype):
        # Normalize parameters
        variant = None if variant == "" else variant

        controlnet_model = ControlNetUnionModel.from_pretrained(
            model_id, variant=variant, torch_dtype=dtype
        )

        components.add(f"controlnet_{self.node_id}", controlnet_model)
        return {
            "controlnet_union_model": components.get_model_info(
                f"controlnet_{self.node_id}"
            )
        }


class UnetLoader(NodeBase):
    def __del__(self):
        components.remove(f"unet_{self.node_id}")
        super().__del__()

    def execute(self, model_id, subfolder, variant, dtype):
        # Normalize parameters
        subfolder = None if subfolder == "" else subfolder
        variant = None if variant == "" else variant

        logger.debug(
            f" load unet from model_id: {model_id}, subfolder: {subfolder}, variant: {variant}, dtype: {dtype}"
        )

        # for the searching and autocomplete later, we download the model_index.json if it has one
        try:
            _config_file = hf_hub_download(model_id, filename="model_index.json")
        except EntryNotFoundError as _e:
            logger.info(
                "Unet model doesn't have a main model_index.json, model won't appear in autocomplete"
            )

        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder=subfolder, variant=variant, torch_dtype=dtype
        )
        components.add(f"unet_{self.node_id}", unet)
        return {"unet": components.get_model_info(f"unet_{self.node_id}")}


class VAELoader(NodeBase):
    def __del__(self):
        components.remove(f"vae_{self.node_id}")
        super().__del__()

    def execute(self, model_id, subfolder, variant, dtype):
        # Normalize parameters
        subfolder = None if subfolder == "" else subfolder
        variant = None if variant == "" else variant

        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder=subfolder, variant=variant, torch_dtype=dtype
        )
        components.add(f"vae_{self.node_id}", vae)
        return {"vae": components.get_model_info(f"vae_{self.node_id}")}


class ModelsLoader(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        # lora
        lora_step = StableDiffusionXLLoraStep()
        self._lora_node = ModularPipeline.from_block(lora_step)
        # ip adapter
        ip_adapter_block = StableDiffusionXLIPAdapterStep()
        self._ip_adapter_node = ModularPipeline.from_block(ip_adapter_block)

    def __del__(self):
        components.remove(f"text_encoder_{self.node_id}")
        components.remove(f"text_encoder_2_{self.node_id}")
        components.remove(f"tokenizer_{self.node_id}")
        components.remove(f"tokenizer_2_{self.node_id}")
        components.remove(f"scheduler_{self.node_id}")
        components.remove(f"unet_{self.node_id}")
        components.remove(f"vae_{self.node_id}")
        components.remove(f"image_encoder_{self.node_id}")
        components.remove(f"feature_extractor_{self.node_id}")
        self._lora_node.unload_lora_weights()
        self._ip_adapter_node.unload_ip_adapter()
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
        ip_adapter_input=None,
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
            in SDXLModelsLoader execute: {self.node_id}
            old_params: {self._old_params}
            new params:
            - repo_id: {repo_id}
            - variant: {variant}
            - dtype: {dtype}
            - unet: {unet}
            - vae: {vae}
            - lora_list: {lora_list}
            ip_adapter_input: {ip_adapter_input}
        """)

        repo_changed = _has_changed(
            self._old_params, {"repo_id": repo_id, "variant": variant, "dtype": dtype}
        )
        unet_input_changed = _has_changed(self._old_params, {"unet": unet})
        vae_input_changed = _has_changed(self._old_params, {"vae": vae})
        lora_input_changed = _has_changed(self._old_params, {"lora_list": lora_list})
        ip_adapter_input_changed = _has_changed(
            self._old_params, {"ip_adapter_input": ip_adapter_input}
        )

        logger.debug(
            f"Changes detected - repo: {repo_changed}, unet: {unet_input_changed}, vae: {vae_input_changed}, "
            f"lora: {lora_input_changed}, ip_adapter: {ip_adapter_input_changed}"
        )

        unet_changed = unet_input_changed or (unet is None and repo_changed)
        vae_changed = vae_input_changed or (vae is None and repo_changed)

        # Load and update base models
        if unet is None:
            if repo_changed or unet_input_changed:
                print(
                    f" load unet from repo_id: {repo_id}, subfolder: unet, variant: {variant}, dtype: {dtype}"
                )
                unet = UNet2DConditionModel.from_pretrained(
                    repo_id, subfolder="unet", variant=variant, torch_dtype=dtype
                )
                # Only add to components if we loaded it ourselves
                print(f" add unet to components: unet_{self.node_id}")
                components.add(f"unet_{self.node_id}", unet)
            else:
                # Get existing unet from our components
                unet = components.get(f"unet_{self.node_id}")
            unet_model_id = (
                f"unet_{self.node_id}"  # Always use our node_id if input is None
            )
        else:
            # unet is always a model info dict if not None
            unet_model_id = unet["model_id"]
            unet = components.get(unet_model_id)

        if vae is None:
            if repo_changed or vae_input_changed:
                logger.debug(
                    f" load vae from repo_id: {repo_id}, subfolder: vae, variant: {variant}, dtype: {dtype}"
                )
                vae = AutoencoderKL.from_pretrained(
                    repo_id, subfolder="vae", variant=variant, torch_dtype=dtype
                )
                # Only add to components if we loaded it ourselves
                logger.debug(f" add vae to components: vae_{self.node_id}")
                components.add(f"vae_{self.node_id}", vae)
            else:
                # Get existing vae from our components
                vae = components.get(f"vae_{self.node_id}")
            vae_model_id = (
                f"vae_{self.node_id}"  # Always use our node_id if input is None
            )
        else:
            # vae is always a model info dict if not None
            vae_model_id = vae["model_id"]
            vae = components.get(vae_model_id)

        # Load text encoders and scheduler
        if repo_changed:
            logger.debug(
                f" load text encoders and scheduler from pipeline: {repo_id}, variant: {variant}, dtype: {dtype}"
            )
            pipe = DiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                variant=variant,
                vae=None,
                unet=None,
            )
            components.add(f"text_encoder_{self.node_id}", pipe.text_encoder)
            components.add(f"text_encoder_2_{self.node_id}", pipe.text_encoder_2)
            components.add(f"tokenizer_{self.node_id}", pipe.tokenizer)
            components.add(f"tokenizer_2_{self.node_id}", pipe.tokenizer_2)
            components.add(f"scheduler_{self.node_id}", pipe.scheduler)

        # Handle LoRA
        if not lora_list:
            logger.debug(f" unload lora from components: lora_{self.node_id}")
            self._lora_node.unload_lora_weights()
        elif lora_input_changed or unet_changed:
            logger.debug(f" update lora from components: lora_{self.node_id}")

            # Unload first to clean previous model's state
            self._lora_node.unload_lora_weights()

            self._lora_node.update_states(
                unet=unet,  # Use actual model
                text_encoder=components.get(f"text_encoder_{self.node_id}"),
                text_encoder_2=components.get(f"text_encoder_2_{self.node_id}"),
            )
            update_lora_adapters(self._lora_node, lora_list)

        # Handle IP-Adapter
        if not ip_adapter_input:
            if self._ip_adapter_node is not None:
                logger.debug(
                    f" unload ip_adapter from components: ip_adapter_{self.node_id}"
                )
                self._ip_adapter_node.unload_ip_adapter()
        elif ip_adapter_input_changed or unet_changed:
            # Unload first to clean previous model's state
            logger.debug(" unload ip_adapter")
            self._ip_adapter_node.unload_ip_adapter()

            if ip_adapter_input_changed:
                logger.debug(
                    f" load image_encoder from repo_id: {ip_adapter_input['repo_id']}, subfolder: {ip_adapter_input['image_encoder_path']}, dtype: {dtype}"
                )
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    ip_adapter_input["repo_id"],
                    subfolder=ip_adapter_input["image_encoder_path"],
                    torch_dtype=dtype,
                )
                feature_extractor = CLIPImageProcessor(size=224, crop_size=224)
                logger.debug(
                    f" add image_encoder to components: image_encoder_{self.node_id}"
                )
                components.add(f"image_encoder_{self.node_id}", image_encoder)
                logger.debug(
                    f" add feature_extractor to components: feature_extractor_{self.node_id}"
                )
                components.add(f"feature_extractor_{self.node_id}", feature_extractor)
                logger.debug(
                    f" update ip_adapter from components: ip_adapter_{self.node_id}"
                )
                self._ip_adapter_node.update_states(
                    image_encoder=image_encoder, feature_extractor=feature_extractor
                )

            if unet_changed:
                logger.debug("update unet")
                self._ip_adapter_node.update_states(unet=unet)  # Use actual model

            logger.debug(" load ip_adapter")
            self._ip_adapter_node.load_ip_adapter(
                ip_adapter_input["repo_id"],
                subfolder=ip_adapter_input["subfolder"],
                weight_name=ip_adapter_input["weight_name"],
            )
            # Set the scale after loading
            self._ip_adapter_node.set_ip_adapter_scale(ip_adapter_input["scale"])

        if unet_changed or vae_changed or repo_changed:
            components.enable_auto_cpu_offload(device=device)

        # Construct loaded_components at the end after all modifications
        loaded_components = {
            "unet_out": components.get_model_info(unet_model_id),
            "vae_out": components.get_model_info(vae_model_id),
            "text_encoders": {
                k: components.get_model_info(v)
                for k, v in {
                    "text_encoder": f"text_encoder_{self.node_id}",
                    "text_encoder_2": f"text_encoder_2_{self.node_id}",
                    "tokenizer": f"tokenizer_{self.node_id}",
                    "tokenizer_2": f"tokenizer_2_{self.node_id}",
                }.items()
            },
            "scheduler": components.get_model_info(f"scheduler_{self.node_id}"),
            "ip_adapter": {
                "image_encoder": components.get_model_info(
                    f"image_encoder_{self.node_id}"
                ),
                "feature_extractor": components.get_model_info(
                    f"feature_extractor_{self.node_id}"
                ),
                "unet": components.get_model_info(unet_model_id, fields="model_id"),
            }
            if ip_adapter_input
            else None,
        }

        logger.debug(f" Final components state: {components}")
        return loaded_components


class EncodePrompt(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        text_block = StableDiffusionXLTextEncoderStep()
        self._text_encoder_node = ModularPipeline.from_block(text_block)

    def execute(self, text_encoders, **kwargs):
        text_encoder_components = {
            "text_encoder": components.get(text_encoders["text_encoder"]["model_id"]),
            "text_encoder_2": components.get(
                text_encoders["text_encoder_2"]["model_id"]
            ),
            "tokenizer": components.get(text_encoders["tokenizer"]["model_id"]),
            "tokenizer_2": components.get(text_encoders["tokenizer_2"]["model_id"]),
        }

        self._text_encoder_node.update_states(**text_encoder_components)
        text_state = self._text_encoder_node(**kwargs)
        # Return the intermediates dict instead of the PipelineState
        return {"embeddings": text_state.intermediates}


class Scheduler(NodeBase):
    def execute(self, input_scheduler, scheduler, **kwargs):
        scheduler_component = components.get(input_scheduler["model_id"])

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

        new_scheduler = scheduler_cls.from_config(
            scheduler_component.config, **scheduler_options
        )

        # Using add here overrides the previous scheduler if there's one
        components.add(f"scheduler_{self.node_id}", new_scheduler)

        return {
            "output_scheduler": components.get_model_info(f"scheduler_{self.node_id}")
        }


class Denoise(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        sdxl_auto_blocks = SDXLAutoBlocks()
        self._denoise_node = ModularPipeline.from_block(sdxl_auto_blocks)
        self._num_inference_steps = 0

    def step_progress_update(self, _pipe, step, _timestep, data):
        if self.node_id:
            try:
                progress = int((step + 1) / self._num_inference_steps * 100)
                asyncio.run_coroutine_threadsafe(
                    web_server.client_queue.put(
                        {
                            "client_id": self._client_id,
                            "data": {
                                "type": "progress",
                                "nodeId": self.node_id,
                                "progress": progress,
                            },
                        }
                    ),
                    web_server.event_loop,
                )
            except Exception as e:
                logger.warning(f"Error queuing progress update: {str(e)}")

        return data

    def execute(
        self,
        unet,
        scheduler,
        embeddings,
        steps,
        cfg,
        seed,
        width,
        height,
        guider,
        controlnet,
        ip_adapter_image_embeddings=None,
    ):
        unet_component = components.get(unet["model_id"])
        scheduler_component = components.get(scheduler["model_id"])

        self._denoise_node.update_states(
            unet=unet_component, scheduler=scheduler_component
        )

        generator = torch.Generator(device="cpu").manual_seed(seed)
        self._num_inference_steps = steps

        denoise_kwargs = {
            **embeddings,  # Now embeddings is already a dict
            "generator": generator,
            "guidance_scale": cfg,
            "height": height,
            "width": width,
            "output": "latents",
            "num_inference_steps": self._num_inference_steps,
            "callback_on_step_end": self.step_progress_update,
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
                    components.get(model_id) for model_id in model_ids
                ]
                controlnet_components = MultiControlNetModel(controlnet_components)
            else:
                controlnet_components = components.get(model_ids)

            self._denoise_node.update_states(controlnet=controlnet_components)

        if guider is not None:
            self._denoise_node.update_states(guider=guider["guider"])
            denoise_kwargs["guider_kwargs"] = guider["guider_kwargs"]
        else:
            # reset the guider in case if it was set before
            self._denoise_node.update_states(guider=CFGGuider())
            denoise_kwargs["guider_kwargs"] = None

        latents = self._denoise_node(**denoise_kwargs)
        logger.debug(f" Components after Denoise: {components}")
        return {"latents": latents}


class DecodeLatents(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        decoder_block = StableDiffusionXLAutoDecodeStep()
        self._decoder_node = ModularPipeline.from_block(decoder_block)

    def execute(self, vae, latents):
        vae_component = components.get(vae["model_id"])
        self._decoder_node.update_states(vae=vae_component)
        images_output = self._decoder_node(latents=latents, output="images")
        return {"images": images_output.images}


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
    def execute(self, pag_scale, pag_layers):
        # TODO: Maybe do some validations to ensure correct layers
        layers = [item.strip() for item in pag_layers.split(",")]
        guider = PAGGuider(pag_applied_layers=layers)
        guider_kwargs = {"pag_scale": pag_scale}
        return {"guider": {"guider": guider, "guider_kwargs": guider_kwargs}}


class APGOptionalGuider(NodeBase):
    def execute(self, momentum, rescale_factor):
        guider = APGGuider()
        guider_kwargs = {"momentum": momentum, "rescale_factor": rescale_factor}
        return {"guider": {"guider": guider, "guider_kwargs": guider_kwargs}}


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


class IPAdapter(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)
        ip_adapter_block = StableDiffusionXLIPAdapterStep()
        self._ip_adapter_node = ModularPipeline.from_block(ip_adapter_block)
        self._last_embeddings = None  # Store last valid embeddings

    def __del__(self):
        components.remove(f"image_encoder_{self.node_id}")
        components.remove(f"feature_extractor_{self.node_id}")
        self._ip_adapter_node.unload_ip_adapter()
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

    def execute(self, unet, ip_adapter_inputs):
        new_params = {"unet": unet, "ip_adapter_inputs": ip_adapter_inputs}
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
            print(f" unload ip_adapter from components: ip_adapter_{self.node_id}")
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

        need_process = False

        if ip_adapter_config_changed:
            # Load the first image encoder (they should all be the same)
            logger.debug(
                f" load image_encoder from repo_id: {repo_ids[0]}, subfolder: {subfolders[0]}"
            )
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                repo_ids[0],
                subfolder=image_encoder_paths[0],
                torch_dtype=torch.float16,  # TODO: Make configurable
            )
            feature_extractor = CLIPImageProcessor(size=224, crop_size=224)
            logger.debug(
                f" add image_encoder to components: image_encoder_{self.node_id}"
            )
            components.add(f"image_encoder_{self.node_id}", image_encoder)
            logger.debug(
                f" add feature_extractor to components: feature_extractor_{self.node_id}"
            )
            components.add(f"feature_extractor_{self.node_id}", feature_extractor)
            need_process = True
        else:
            image_encoder = components.get(f"image_encoder_{self.node_id}")
            feature_extractor = components.get(f"feature_extractor_{self.node_id}")

        if unet_changed or ip_adapter_config_changed:
            logger.debug(" update ip_adapter state")
            self._ip_adapter_node.unload_ip_adapter()
            self._ip_adapter_node.update_states(
                unet=components.get(unet["model_id"]),
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )

            logger.debug(" load ip_adapter(s)")
            self._ip_adapter_node.load_ip_adapter(
                repo_ids,
                subfolder=subfolders,
                weight_name=weight_names,
            )
            need_process = True

        if ip_adapter_scale_changed:
            logger.debug(f" set ip_adapter scale: {scale}")
            self._ip_adapter_node.set_ip_adapter_scale(scale)

        if ip_adapter_image_changed or need_process:
            logger.debug(" process ip_adapter image")
            ip_adapter_state = self._ip_adapter_node(ip_adapter_image=image)
            self._last_embeddings = ip_adapter_state.intermediates

        return {
            "ip_adapter_image_embeddings": self._last_embeddings,
            "unet_out": components.get_model_info(unet["model_id"]),
        }
