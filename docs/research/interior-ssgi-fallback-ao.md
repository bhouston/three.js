# Fallback-only diffuse AO experiment

Base: `8de0e0315c5c`. Toggle **Occlusion → Diffuse AO scope** between **All indirect** and **Fallback only** (experimental default).

Fallback-only applies screen AO to fallback providers before resolving their coverage. Already visibility-integrated SSGI is not multiplied by GTAO again. Material AO remains active. The specular occlusion hook receives the original combined material/screen visibility, retaining the existing GTAO+GTSO treatment. No inverse-AO amplification is used. Transparent materials keep their existing AO exclusion.

This intentionally retains AO on baked irradiance for comparison; fully resolved lightmaps may already contain that visibility and need no extra AO. Compare None, All indirect, and Fallback only against independent reference, especially Cornell corners and the Dual Rooms doorway. The experiment addresses screen AO on diffuse transport; sheen uses the engine's scalar ambient AO and would require separate lobe accounting. The no-bent-normal GTAO mode reconstructs the base specular AO approximation in the context hook; layered clearcoat needs further validation.

Lint passed. Central sequential GPU validation is pending. This branch is an experiment, not a validated default.
