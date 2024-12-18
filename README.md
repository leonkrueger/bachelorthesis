# Content
- .env: Used to specify the environment variables used in this project. All variables that could be required are present.
- evaluation.slurm: Used to start a slurm-job that executes bachelorthesis/evaluation.py. (requires a virutal environment in venv/ with modules from requirements.txt)
- fine_tuning.slurm: Used to start a slurm-job that executes bachelorthesis/fine_tuning.py. (requires a virutal environment in venv/ with modules from requirements.txt)

Python scripts in subfolder bachelorthesis (further explanation at the beginning of each script):
- main.py: Manually test the system
- fine_tuning.py: Fine-tune a model for this task
- evaluation.py: Evaluate the system with the main evaluation task
- evaluate_table_prompt_output.py: Evaluate individual prompt outputs of the table prediction model
- evaluate_column_prompt_output.py: Evaluate individual prompt outputs of the columns mapping model
- synonym_generation.py: Generate synonyms used in the evaluation task


# Fine-tuned models
The best working models for both tasks were:
- missing_tables_12000_1_csv_data_collator
- missing_columns_12000_1_own_data_collator

These were also the models I used in the evaluation of my thesis.