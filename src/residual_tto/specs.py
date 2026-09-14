from dataclasses import dataclass

@dataclass(frozen=True)
class InterventionSpec:
    name: str
    site: str
    layer: int
    view_ids: tuple
    token_scope: str
    register_ids: tuple = ()

    def slots(self):
        table = {"camera": (0,), "registers": tuple(range(1, 17)),
                 "camera_registers": tuple(range(17))}
        if self.token_scope == "selected_registers":
            ids = self.register_ids
            if not ids or len(set(ids)) != len(ids) or any(i not in range(16) for i in ids):
                raise ValueError("register_ids must be unique, nonempty, and in [0,15]")
            return tuple(1 + i for i in ids)
        if self.token_scope not in table:
            raise ValueError("invalid token_scope")
        return table[self.token_scope]

    def validate(self, views, layers=24):
        if self.site not in ("pre_frame", "post_frame") or self.layer not in range(layers):
            raise ValueError("invalid injection site")
        if not self.name or not self.view_ids or len(set(self.view_ids)) != len(self.view_ids):
            raise ValueError("name and unique editable views required")
        if any(i not in range(views) for i in self.view_ids):
            raise ValueError("editable view out of range")
        self.slots()

def resolve_specs(config, camera_order):
    if len(camera_order) != len(set(camera_order)):
        raise ValueError("duplicate camera identity")
    specs = []
    occupied = set()
    for item in config:
        spec = InterventionSpec(item["name"], item["site"], item["layer"],
            tuple(camera_order.index(n) for n in item["editable_views"]),
            item["token_scope"], tuple(item.get("register_ids", [])))
        spec.validate(len(camera_order))
        for view in spec.view_ids:
            for slot in spec.slots():
                key = (spec.site, spec.layer, view, slot)
                if key in occupied:
                    raise ValueError("overlapping interventions are not permitted")
                occupied.add(key)
        specs.append(spec)
    if not specs or len({s.name for s in specs}) != len(specs):
        raise ValueError("unique nonempty specs required")
    return specs
