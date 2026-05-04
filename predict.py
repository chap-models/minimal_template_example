import argparse
import joblib
import pandas as pd


def predict(model_fn: str, historic_data: str, future_data: str, out_file: str):
    # chap eval supplies a future CSV without `disease_cases` (the target isn't
    # known yet), so we can't read y_val from it. The previous version computed
    # an RMSE return value that was never captured by chap eval anyway.
    df = pd.read_csv(future_data)
    X = df[['rainfall', 'mean_temperature']]
    model = joblib.load(model_fn)
    df['sample_0'] = model.predict(X)
    df.to_csv(out_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')
    parser.add_argument('model_config', type=str, help='Path to model configuration yaml.') # not used not sure why needed

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)