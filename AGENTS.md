# Project workflow

- Before any cluster or GPU work, read and follow `00_reference/cluster_gpu_workflow.md`. Use the project's actual environment; activate environments and submit jobs only within the user-authorized scope. Never train on the login node, use another owner's allocated GPUs, cancel unrelated jobs, or bypass access restrictions.

- Follow `00_reference/project_plan.md` as a changeable working plan. Do not silently expand the project scope.
- Work incrementally on the user's requested step. The user has now authorized setup of the external `env-speech` environment and small STRF/dSTRF experiments, starting with the Wingert et al. CNN method. Follow cluster restrictions for training.
- Track actual completed work and the next step in `00_reference/progress.md`. Mark experiments complete only when they have actually run and their outputs have been checked.
- After running experiments, add a concise, readable Markdown summary in `05_results/README.md` with settings, main findings, limitations, and output paths. Create separate result documents only when needed; avoid many small or redundant reports.
- Keep the plan, progress, and observed results distinct. Never present planned or expected results as measurements.
- Maintain dependencies in `00_reference/requirements.txt`. Use `/clusterfs/bebb/users/tonychae01/envs/env-speech`; keep the environment outside this repository.
- Keep code concise. Use notebooks with short English explanations for interactive understanding, and `03_src/pipeline/` for repeatable or batch runs. All repository documentation should be in English.
- Keep raw data, processed data, source, checkpoints, results, and external resources in their corresponding numbered folders.
