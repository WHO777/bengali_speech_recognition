import enum
from types import MappingProxyType


class Language(enum.Enum):
    RUSSIAN = enum.auto()
    ENGLISH = enum.auto()


RUSSIAN_VOCABULARY = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя'!?"
ENGLISH_VOCABULARY = "abcdefghijklmnopqrstuvwxyz'!?"

STR_LANGUAGE_TO_ENUM_MAP = MappingProxyType({
    'russian': Language.RUSSIAN,
    'english': Language.ENGLISH,
})

LANGUAGE_TO_VOCABULARY_MAP = MappingProxyType({
    Language.RUSSIAN:
    tuple(RUSSIAN_VOCABULARY),
    Language.ENGLISH:
    tuple(ENGLISH_VOCABULARY),
})
