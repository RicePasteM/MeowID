# MeowID project page

A dependency-free static project page for the paper “MeowID: A Dual-Expert Retrieval System for Individual Cat Identification.”

## Preview locally

Open `index.html` directly, or run a tiny local server from the `project_page` directory:

```bash
cd /Users/ricepastem/Desktop/MeowID/project_page
python3 -m http.server 8080
```

Then visit `http://localhost:8080`.

All visual assets, the paper PDF, and the locally vendored Lucide icon library are bundled under `assets/`; no build step or network connection is required for the page itself.
