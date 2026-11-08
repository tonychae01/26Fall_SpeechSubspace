# Project workflow

- Follow `00_reference/project_plan.md` as a changeable working plan. Do not silently expand the project scope.
- Work incrementally on the user's requested step. Initial setup authorizes folders and documentation only; environment setup and experiments are deferred.
- Track actual completed work and the next step in `00_reference/progress.md`. Mark experiments complete only when they have actually run and their outputs have been checked.
- After running experiments, add a concise, readable Markdown summary in `05_results/README.md` with settings, main findings, limitations, and output paths. Create separate result documents only when needed; avoid many small or redundant reports.
- Keep the plan, progress, and observed results distinct. Never present planned or expected results as measurements.
- Add dependencies to `00_reference/requirements.txt` as needed later. The virtual environment will be outside this repository in an `envs` directory; do not create it during initial setup.
- Keep raw data, processed data, source, checkpoints, results, and external resources in their corresponding numbered folders.
