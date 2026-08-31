import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
deps = data.get('project', {}).get('dependencies', [])
with open('requirements.in', 'w') as f:
    for dep in deps:
        f.write(dep + '\n')
