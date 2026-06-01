# Contributing

We welcome external contributions from the open-source community.
[Open an issue](https://github.com/feedzai/fairgbm/issues) to report problems or recommend new features.


## Development Setup

Install the package in editable mode with dev dependencies (builds the native extension and places it automatically):

```bash
pip install -e ".[dev]"
```

This handles both compiling the C++ extension and installing test dependencies (hypothesis, pytest, etc.).

### Manual build (alternative)

If you prefer to build the native extension separately:

```bash
cd native
mkdir -p build && cd build
cmake .. && make -j4
```

The compiled `lib_fairgbm.so` will be placed directly into the `fairgbm/` package directory.

### Running tests

```bash
pytest tests/
```


## Contributor List

- André Cruz
- Catarina Belém
- Sérgio Jesus
- Gonçalo Arsénio
- Alberto Ferreira
- João Veiga
- Sara Guerreiro
- Pedro Gandola
- João Bravo
- Pedro Saleiro
- Pedro Bizarro

We also thank all [LightGBM contributors](https://github.com/microsoft/LightGBM/graphs/contributors) as the FairGBM 
project was initially built upon that code-base.
