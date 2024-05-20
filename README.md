# ML Pytorch Project Template

## Config
Using template files provided and changing data to corresponding data you can use repository.

`train.py` takes one argument of name config which is yaml file which template is placed in templates directory.

## Using config file
Copy config file to main directory:
```bash
cp templates/config.yaml.template config.yaml
```

Then insert your required credentials

## Pylint
For linux bash pylint is supported via command:
```bash
bash pylint.sh
```

Either way add pylint to IDE or editor to work with project without failing on push.