# How to build apptainer

Optionally, export the APPTAINER_TMPDIR in order to use it as temporary working space when building.
Temporary space is also used when running containers in unprivileged mode, and performing some operations
on filesystems that do not fully support `--fakeroot`.

```Bash
export APPTAINER_TMPDIR=/scratch/apptainer/$USER
```

To build the `test.sandbox` from definition file use `dbgc.def`:

```Bash
apptainer build --sandbox --fix-perms --fakeroot test.sandbox dbgc.def
```

Optionally, one can modify the `test.sandbox` before freezing it into `sif` container file.

```Bash
apptainer shell --writable --fakeroot --nv --bind <local_path>:<remote_path>:<opts> test.sandbox/
```

Finally, create apptainer image:

```Bash
apptainer build dbgc.sif test.sandbox/
```

Do not forget to remove the `test.sandbox` in the end as it takes lots of storage space:

```Bash
rm -rf test.sandbox
```

The build assumes that the `PROJECT_ROOT` is located at "$HOME/work/fictional-pancake"
To run, for example training

```Bash
apptainer run --bind <ABSOLUTE_PATH_TO_HOME>:<ABSOULTE_PATH_TO_HOME> --nv dbgc.sif src/train.py +experiment@_here_=exp_train_mdg dataset_name=unittest_dataset paths.data_dir=$HOME/data
```

This is done to make it easier to modify the project and configs. If there are no frequent changes, it would be better to bake the project files in the container,
as this reduces the amount of traffic between the nodes during the execution on the cluster.
