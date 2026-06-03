from enum import Enum


class AnnotationMode(str, Enum):
    MANUAL = "manual"
    ASSIST = "assist"
    AUTO = "auto"
    FULL_AUTO = "full-auto"
