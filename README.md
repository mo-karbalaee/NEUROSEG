# NEUROSEG

This is a guide for running NEUROSEG locally. 

## Fetching Dependencies

Run the following command in the root directory to fetch the dependencies. 

```shell
uv sync
```

## Running 

In order to run NEUROSEG, you only need to run the following command:

```shell
uv run -m neuroseg.main
```

Put the recordings that you want to process in the `data` folder.
After running the pipeline, you can find the masks and activity trace 
pictures in the `output` folder. 
