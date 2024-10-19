from args import cmd_args
import os

if __name__ == '__main__':
    fname = cmd_args.file.split('/')[1]
    
    if fname == 'main_anomaly.py':
        from anomaly.main_anomaly import main 
    elif fname == 'baseline_main.py':
        from baselines.baseline_main import main 
    elif fname == 'baseline_main_feat.py':
        from baselines.baseline_main_feat import main 
    elif fname == 'greedy_main.py':
        from baselines.greedy_main import main 
    elif fname == 'main.py':
        from src.main import main 
    elif fname == 'main_feat.py':
        from src.main_feat import main 
    elif fname == 'main_online.py':
        from src.main_online import main 
    main(cmd_args)
    