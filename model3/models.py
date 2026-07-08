from typing import Any


class BrandMappingResult:
    def __init__(
        self,
        input_name: str = "",
        brand_name: str = "",
        generic_name: str = "",
        active_ingredients: list[str] | None = None,
        strength: str = "",
        dosage_form: str = "",
        manufacturer: str = "",
        smiles: str | None = None,
        ndc: str = "",
        source: str = "openFDA",
        cached: bool = False,
    ) -> None:
        self.input_name = input_name
        self.brand_name = brand_name
        self.generic_name = generic_name
        self.active_ingredients = active_ingredients or []
        self.strength = strength
        self.dosage_form = dosage_form
        self.manufacturer = manufacturer
        self.smiles = smiles
        self.ndc = ndc
        self.source = source
        self.cached = cached

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_name": self.input_name,
            "brand_name": self.brand_name,
            "generic_name": self.generic_name,
            "active_ingredients": self.active_ingredients,
            "strength": self.strength,
            "dosage_form": self.dosage_form,
            "manufacturer": self.manufacturer,
            "smiles": self.smiles,
            "ndc": self.ndc,
            "source": self.source,
            "cached": self.cached,
        }
