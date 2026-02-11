from __future__ import annotations

from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset, Features, Value
from typing import Optional, Self, get_args, get_origin, Union

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


def _unwrap_optional(annotation):
    """
    If annotation is Optional[T] / Union[T, None], return (T, True).
    Otherwise return (annotation, False).
    """
    origin = get_origin(annotation)

    # Optional[T] is Union[T, NoneType] at runtime
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return non_none[0], True

    return annotation, False


def _py_type_to_hf_value(py_type) -> Value:
    """
    Map a python (possibly unwrapped) type to a Hugging Face datasets Value dtype.
    Extend this as needed if you add more complex fields (lists, dicts, etc.).
    """
    # Handle forward refs / string annotations if any appear
    if isinstance(py_type, str):
        # Best effort: treat unknown forward refs as string
        return Value("string")

    # Basic scalar mappings
    if py_type is str:
        return Value("string")
    if py_type is int:
        return Value("int64")
    if py_type is float:
        return Value("float64")
    if py_type is bool:
        return Value("bool")

    # If you later add Enums, you can map to string:
    # if isinstance(py_type, type) and issubclass(py_type, Enum): return Value("string")

    raise TypeError(f"Unsupported field type for HF Features mapping: {py_type!r}")


def features_from_pydantic_model(model_cls: type[BaseModel]) -> Features:
    """
    Build a Hugging Face Features schema from a Pydantic v2 model.

    Notes:
    - HF `Value(...)` columns can contain nulls, so Optional fields don't need special handling.
    - This function maps only scalar fields (str/int/float/bool). Extend if you add lists/dicts.
    """
    feature_map = {}
    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        base_type, _is_optional = _unwrap_optional(annotation)
        feature_map[name] = _py_type_to_hf_value(base_type)

    return Features(feature_map)


class Item(BaseModel):
    """
    An Item is a data-point of a Product with a Price
    """

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str):
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push Item lists to HuggingFace Hub with a stable, model-derived schema"""
        item_features = features_from_pydantic_model(Item)

        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train], features=item_features),
                "validation": Dataset.from_list([item.model_dump() for item in val], features=item_features),
                "test": Dataset.from_list([item.model_dump() for item in test], features=item_features),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load from HuggingFace Hub and reconstruct Items"""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
