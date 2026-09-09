# Inherited legacy code: reference only on scene-adaptation

This directory was inherited from the legacy training branch and is not a new per-scene experiment.
Do not launch or re-enable its PBS scripts from this branch: they point at the sibling legacy checkout.
Do not use its pose/FOV supervised losses as focal-only adaptation. New work belongs in ../experiments/<version>/.
Read ../AGENTS.md and ../README_SCENE_ADAPTATION.md. If legacy training is requested, operate its own checkout.
