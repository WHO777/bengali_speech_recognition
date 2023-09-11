import enum


class Language(enum.Enum):
    RUSSIAN = enum.auto()
    ENGLISH = enum.auto()


LANGUAGE_TO_VOCABULARY_MAP = {
    Language.RUSSIAN: [s for s in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя'!?"],
    Language.ENGLISH: [s for s in "abcdefghijklmnopqrstuvwxyz'!?"]
}