# Spatial Tests

Tests in this folder cover spatial policy and CORINE buffer-confidence
behavior.

They use small synthetic geometries to verify:

- pure buffers produce point-label share 1;
- boundary buffers produce area-weighted mixed shares;
- dominant labels can differ from point labels;
- artificial CORINE classes are retained;
- no-intersection buffers return safe null/zero values;
- entropy and normalized entropy behave as expected.
