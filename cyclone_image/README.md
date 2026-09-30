# Model 1 Input Data

Place the locally acquired `Cyclone_Images.h5` file in this directory. It must contain an `Images` dataset shaped `(N, 128, 128, 4)` with four real spectral bands. Large image archives and annotation masks are not tracked in Git.

Model 1 training also requires `image_annotations.csv` here. The annotation schema, storm-disjoint split rules, and mask format are documented in the repository's [project guide](../CYCLONEVISION_PROJECT_EXPLANATION_GUIDE.md).
