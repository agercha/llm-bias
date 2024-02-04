In order to run experiments, just use
python3.8 replacement_individual.py

This will measure improvements on all category prompts pairs individually.

Alternatively, run
python3.8 test_completions.py

This will randomly select a brand and category and prompt and record completions for them. It will run in an infinite loop, so make sure to interrupt it eventually.

These results can be analyzed with stabilize_results.py and graph_improvments.py