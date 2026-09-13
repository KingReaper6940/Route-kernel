# Showcase website and Vercel deployment

The website lives in `site/`. Its npm manifest and `vercel.json` live at the Git
repository root, next to `pyproject.toml`. Vercel builds the static website with
Vite; Python, CUDA, model downloads, API keys, and paid GPU resources are not
part of the web deployment.

## Import into Vercel

1. Commit and push the website files, `package-lock.json`, and the recorded
   `docs/results/nvidia-a40/` evidence to GitHub.
2. In Vercel, choose **Add New → Project** and import `Route-kernel`.
3. Keep **Root Directory** at the repository root (`./`). The local parent
   folder named `RouTe kernel` is not part of the Git repository.
4. Deploy. `vercel.json` selects Vite, runs `npm ci` then `npm run build`,
   and serves `site-dist`. No environment variables are required.

Do not select `site/` as the Vercel root: the build reads recorded results from
`docs/`. The production build is tested locally; an actual Vercel deployment
requires importing the pushed repository into your account.

## Local development

Use Node 22.12 or newer (Node 22 LTS is suitable):

```bash
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. For the production artifact:

```bash
npm run build
npm run preview
```

Open `http://127.0.0.1:4173`. These commands work from PowerShell too; use
`npm.cmd` if local PowerShell execution policy blocks `npm.ps1`.

## Evidence and behavior

`scripts/prepare-site.mjs` reads the eight committed NVIDIA sweep reports,
checks correctness status, derives the headline statistics, and copies a
specific allowlist of evidence into `site/public/evidence/`. These generated
files are ignored by Git and recreated on every build. No local environment,
SSH key, account settings, or rental archive is copied into the deployment.

The benchmark selector uses measured values, including the regression. The
routing diagram uses a deterministic JavaScript grouping simulation. It does
not call a model or perform a GPU benchmark in the browser. Fonts are served
with the website, without requests to a third-party font CDN. The page includes
keyboard focus, a skip link, native form controls, and reduced-motion support.

## Checks

```bash
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

Browser tests cover desktop/mobile layouts, routing controls, benchmark
selection, regression wording, p95 values, evidence downloads, keyboard use,
clipboard feedback, and horizontal overflow. Screenshots are produced under
`test-results/`. `.github/workflows/website.yml` runs the same checks on pushes
and pull requests; local checks do not imply that hosted CI has already run.

Deployment configuration follows [Vercel's Vite documentation](https://vercel.com/docs/frameworks/frontend/vite).
