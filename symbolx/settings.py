

allowed_string = '.-_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

value_type_name_map = {
                        "v": "Value",
                        "m": "Marginal",
                        "up": "Upper",
                        "lo": "Lower",
                        }


class Settings:
    def __init__(self):
        self.name_value_type_token = {}

    def append(self, name_value_type_token:tuple):
        assert isinstance(name_value_type_token, tuple), "A tuple of symbol name, value_type, and symbol_handler_token must be provided."
        assert name_value_type_token not in self.name_value_type_token, f"Tuple: {name_value_type_token} already exist. Choose a different symbol name."
        self.name_value_type_token[name_value_type_token] = True

    def exists(self, name_value_type_token:tuple):
        assert isinstance(name_value_type_token, tuple), "A tuple of symbol name, value_type, and symbol_handler_token must be provided."
        if name_value_type_token in self.name_value_type_token:
            return True
        else:
            return False

    def getnames(self):
        return self.name_value_type_token

settings = Settings()


# TODO: create a union of all array.coords used sofar. collect all uniques from coo from used arrays.
#       Then compare both to warn which element never used to improve efficiency