# Example assets on Land of Assets

The examples use the public `BenHouston3D/three` project with explicit `1.0.0` versions. `manifest.json` records all 397 uploads, their original sources and filenames, delivery URLs, and SHA256 hashes of the delivered files. Repository assets come from the upstream `dev` revision recorded in the manifest. The dynamic Khronos sample catalogs continue to use their upstream URLs.

The uploads comprise 63 repository models, 4 IES profiles, 18 environment maps, 40 MaterialX packages (28 repository materials and 12 fixed external samples), and 4 fixed external models with 268 progressive LOD dependencies. Original repository assets remain available locally.

## Packaging and delivery

- Land of Assets requires short asset names containing letters, numbers, and single dashes. Upload filenames retain their original names; asset names distinguish collisions and shorten long names.
- The service accepts GLB rather than loose glTF. The 16 repository `.gltf` documents are wrapped in GLB containers with their original buffers and images embedded as data URIs. Scene graphs, extensions, and resource bytes are preserved. Their uploaded filenames retain the original stem with a `.glb` extension.
- MaterialX documents are delivered as `.mtlx.zip` packages containing their resources. Textures are not resized or re-encoded. The CLI rejected the existing rotate3d validation sample, so that one package contains the original XML and texture zipped directly.
- Five UltraHDR JPEG environment maps are stored as `IMAGE` because the service restricts `ENVMAP` to HDR/EXR. Their bytes and example loaders are unchanged.
- Progressive glTF extension paths point to versioned Land of Assets dependency URLs. These URLs end in `.glb` because the progressive loader checks the URL suffix. Other URLs use `?original=true` to request the uploaded format.
- Assets carry source attribution and use the service's `NONE` share-license setting to avoid assigning a new license. Original source licenses continue to apply.

## Verification

Anonymous downloads were checked against the manifest hashes with cross-origin delivery enabled. The 16 GLB containers preserve their source JSON graphs and all 42 embedded resources. Three.js parsed all four IES profiles, twelve HDR maps, and the EXR environment map. All MaterialX archive resources were checked for inclusion and byte preservation. Progressive model graphs and binary chunks were checked for preservation, and their LOD paths were checked for matching resource GUIDs.

GPU rendering was not verified: no browser connection was available during this migration. No software rasterizer was used.
