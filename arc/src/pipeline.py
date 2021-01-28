import inspect


def make(**definitions):
    def register(**new_defs):
        definitions = {**definitions, **new_defs}

    # Whoa!
    register(register=register)

    resolved = {}
    dep_names = {}

    for name, func in definitions.items():
        if callable(func):
            names = list(inspect.signature(func).parameters)
        else:
            names = []

        dep_names[name] = names

    def resolve(name):
        if name in resolved:
            return resolved[name]

        dependencies = {
            dep_name: resolve(dep_name)
            for dep_name in dep_names[name]
        }

        if callable(func):
            resolution = definitions[name](**dependencies)
        else:
            resolution = definitions[name]

        resolved[name] = resolution

        return resolution

    def run(name):
        return resolve(name)

    return run
