#### INSTRUCTIONS

Install [Mellon](https://github.com/cubiq/Mellon/)

clone this repository insde the `custom` directory:

```
cd custom
git clone https://github.com/asomoza/mellon-modular-diffusers.git
```

Uninstall diffusers and install from the latest modular diffusers branch. https://github.com/huggingface/diffusers/pull/11235

```
pip uninstall diffusers -y
pip install git+https://github.com/huggingface/diffusers.git@modular-refactor-0501
```

To use diffusers components manager across different custom nodes, add this in your `custom/__init__.py`


```python
from diffusers import ComponentsManager
components = ComponentsManager()
```

and add this in all your custom nodes module file
```python
from custom import components
```

```
custom/
├── __init__.py              # create your components manager here 
├── mellon-modular-diffusers/
│   └── mellon-modular-diffusers.py # use in this
└── diff-diff/
    └── diff-diff.py # use in this
