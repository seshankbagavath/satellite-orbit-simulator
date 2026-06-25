# Deploying the Web App to GitHub Pages

The interactive app is a single static file (`docs/index.html`) — no build step,
no backend. GitHub Pages can serve it for free.

## One-time setup (≈ 1 minute)

1. Upload the `docs/` folder (containing `index.html`) to your repo, so the path
   is `docs/index.html` on the `main` branch.
2. On GitHub, go to your repo → **Settings** → **Pages** (left sidebar).
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Set **Branch** to `main` and the folder to **`/docs`**, then click **Save**.
5. Wait ~1 minute. GitHub will show a green banner with your live URL:
   `https://seshankbagavath.github.io/satellite-orbit-simulator/`

That URL is already wired into the README's "Live Interactive Demo" link.

## Updating the app later

Just edit `docs/index.html` (or re-upload it) and commit. Pages redeploys
automatically within a minute or two.

## How it works (so you can explain it)

- The orbital math (Keplerian elements → 3D position, J2 nodal regression) is
  ported from `satellite_orbit_simulator.py` into vanilla JavaScript.
- 3D rendering uses **Three.js**, loaded from a CDN — nothing to install.
- The SGP4 validation result is shown as a stat callout plus the
  `fig_val_error.png` image and a link to the notebook, rather than recomputed
  in the browser (SGP4 has no clean JS port). The *interactive* part is the
  live orbit; the *validation* is the credibility proof.

## Notes

- The validation image is loaded from your repo's raw URL, so keep
  `fig_val_error.png` in the repo root.
- If you rename your GitHub username or repo, update the `GH_USER` / `GH_REPO`
  constants near the top of the `<script>` block in `index.html`.
