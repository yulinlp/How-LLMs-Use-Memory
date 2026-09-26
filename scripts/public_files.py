RUNTIME_FILES = ['reproduce.py', 'setup.py', 'requirements-reproduce.txt']
RUNTIME_FILES += ['steem_adapt/' + name + '.py' for name in [
    '__init__', 'io', 'modeling', 'prompts', 'prepare_generation', 'mccs_generate',
    'latent_online_generate', 'reproduction', 'prepare_steem', 'convert_benchmarks']]
RUNTIME_FILES += ['scripts/' + name + '.py' for name in [
    'compass_matrix_generate', 'compass_matrix_reader', 'compass_matrix_judge',
    'duet_executor', 'dac_projection_safe', 'endogenous_memory_gates',
    'reproduction_helpers', 'native_judge_protocol', 'rq3_judge_router',
    'reproduce_rq1', 'analyze_rq1_cross_benchmark', 'build_rq1_reference_split',
    'summarize_reproduction', 'prepare_paper_data',
    'build_steem_latent_unified', 'smoke_reproduction', 'public_files',
    'check_reproduction']]
RUNTIME_FILES += ['tests/test_reproduction.py']
RUNTIME_FILES += ['configs/' + name for name in [
    'paper_sources.json', 'paper_inputs.json', 'paper_splits.json']]

PUBLIC_FILES = RUNTIME_FILES + [
    '.gitignore', 'README.md', 'README_zh.md', 'THIRD_PARTY_NOTICES.md',
    'assets/memory-use-logo-ai.png', 'assets/research-overview.png',
]
