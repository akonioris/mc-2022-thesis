import argparse
from log import get_logger
from loading import load_data
from plotting_tools import pca_clustering
from preprocessing import preprocess
from dimensionality_reduction import pca_clustering
from models_selection import models
from model_deployment import best_model
from optimization import best_model_optimized
from module_parse import models_parse

# This is the main phase of our progect that encapsulates all the functions of our analysis.

parser = argparse.ArgumentParser(description = 'Select the Model to predict Rating')
parser.add_argument('--nomodel', metavar = '', default = True, help = 'Disable model selection')
args = parser.parse_args()

def main_function(args):
    logger = get_logger()
    topcat = load_data(logger)
    df = preprocess(topcat, logger)
    pca_df = pca_clustering(df, logger)
    results_df = models(df, logger)
    outcome = best_model(df, logger)
    result_optimized = best_model_optimized(df, logger)
    parse = models_parse(df, args.nomodel)
    pass
 
if __name__ == '__main__':
    main_function(args)
