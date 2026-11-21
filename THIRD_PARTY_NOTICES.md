# Third-Party Notices For MultiSTEVE-1s Code

This file documents third-party provenance and terms for copied, adapted,
downloaded, or referenced assets in the MultiSTEVE-1s source
release.

## MultiSTEVE-Specific Additions

New MultiSTEVE-specific source additions are released under the MIT Licence in
the top-level `LICENSE` file. This includes the `multisteve` package, release
documentation, plotting suite, configs, wrappers, and source-only helper tools,
except where a file states otherwise.

## Upstream STEVE-1

Repository: https://github.com/Shalev-Lifshitz/STEVE-1

This source release contains inherited or adapted STEVE-1 files, including code
under `steve1/`, `train/`, and `run_agent/`. As of 2026-05-06, the upstream
STEVE-1 repository did not contain a recognised licence file, GitHub reported
`license: null`, and
`https://api.github.com/repos/Shalev-Lifshitz/STEVE-1/license` returned 404.
This release does not grant a new licence for upstream STEVE-1 code, STEVE-1
weights, the STEVE-1 prior, the STEVE-1 prior dataset, or STEVE-1 visual prompt
embeddings.

## OpenAI Video PreTraining

Repository: https://github.com/openai/Video-Pre-Training

This release includes copied or adapted VPT source components under
`steve1/VPT/` and uses VPT model artefacts such as `2x.model` and
`rl-from-foundation-2x.weights` as external downloads. The VPT source repository
is licensed under the MIT Licence:

MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The VPT README publishes public contractor-demonstration indexes and data format
documentation. No separate explicit data licence for those contractor
demonstrations was identified as of 2026-05-06. OpenAI's VPT README states that
Minecraft is Microsoft's intellectual property and that OpenAI is not licensing
Minecraft IP.

## MineCLIP and MineDojo

MineCLIP: https://github.com/MineDojo/MineCLIP

MineDojo: https://github.com/MineDojo/MineDojo

MineCLIP and MineDojo source code are distributed under the MIT Licence.
MineDojo documents separate terms for some knowledge-base components; those
knowledge-base components are not redistributed in this source
release.

## MineRL

Installed fork: https://github.com/KarolisRam/minerl/tree/dev

MultiSTEVE-1s uses Karolis Jucys's fork for creative-mode chat commands.

Upstream repository: https://github.com/minerllabs/minerl

Licence file: https://github.com/minerllabs/minerl/blob/dev/LICENSE

The MineRL environment used for rollouts is distributed under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International. MineRL is an external
dependency and is not vendored in this source repository.

## Minecraft Assets

Minecraft Usage Guidelines:
https://www.minecraft.net/en-us/usage-guidelines

Minecraft is owned by Microsoft/Mojang. Screenshots, videos, game observations,
textures, sounds, and other game-derived media are Minecraft assets under the
Minecraft Usage Guidelines and related policies. This release does not grant any
rights to Minecraft assets, does not redistribute Minecraft game files, and is
not an official Minecraft product. It is not approved by or associated with
Mojang or Microsoft.

## OpenAI Contractor Demonstrations

The training workflows consume OpenAI VPT contractor demonstrations through the
public indexes documented in the VPT repository. The source repository
does not include raw contractor videos. Derived training-data artefacts that
include contractor-derived `frames.mp4`, action, embedding, or metadata files are
not covered by a blanket permissive licence in this source release.

## Hugging Face Model And Dataset Repositories

The MultiSTEVE-1s model zoo, MineCLIP-embedded training data, and seed-image
library are research artefacts derived from upstream VPT, STEVE-1, MineCLIP,
MineDojo, MineRL, OpenAI contractor-demonstration, and Minecraft assets. Those
repositories are marked with `license: other` and the MultiSTEVE-1s research
artefact terms.

## Paper And Visualisation Assets

Twemoji graphics are licensed under CC BY 4.0:
https://github.com/twitter/twemoji/blob/master/LICENSE-GRAPHICS

Plotly.js is licensed under MIT:
https://github.com/plotly/plotly.js/blob/master/LICENSE

Three.js is licensed under MIT:
https://github.com/mrdoob/three.js/blob/dev/LICENSE

Paper icons, generated HTML dashboards, and standalone visualisation bundles
containing these assets retain the relevant upstream notices.

## Other Included Binary Asset

The source release includes
`steve1/data/generation/assets/mouse_cursor_white_16x16.png`. This file is
treated as inherited STEVE-1 material, and no separate author-created licence is
asserted for it.
