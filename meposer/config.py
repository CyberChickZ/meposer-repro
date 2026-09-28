from pathlib import Path

import yaml


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def from_dict(cls, d):
        return cls({k: cls.from_dict(v) if isinstance(v, dict) else v for k, v in d.items()})

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}

    def set_dotted(self, key, value):
        node = self
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), Config):
                node[p] = Config()
            node = node[p]
        node[parts[-1]] = value

    def get_dotted(self, key, default=None):
        node = self
        for p in key.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def merge(self, other):
        for k, v in other.items():
            if isinstance(v, dict) and isinstance(self.get(k), Config):
                self[k].merge(v)
            else:
                self[k] = Config.from_dict(v) if isinstance(v, dict) else v
        return self


def _load_yaml(path):
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("_base_", None)
    cfg = Config()
    if base:
        cfg = _load_yaml(path.parent / base)
    return cfg.merge(raw)


def parse_value(text):
    text = text.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return yaml.safe_load(text)


def apply_overrides(cfg, overrides):
    for item in overrides or ():
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        cfg.set_dotted(key.strip(), parse_value(value))
    return cfg


def load_config(path, overrides=()):
    return apply_overrides(_load_yaml(path), overrides)


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
